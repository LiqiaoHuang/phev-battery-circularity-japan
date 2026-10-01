"""Select AR6 electricity pathways and calculate annual technology shares."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ELECTRICITY = 'Secondary Energy|Electricity'
TECHNOLOGIES = ['Biomass', 'Coal', 'Gas', 'Oil', 'Geothermal', 'Hydro', 'Nuclear',
                'Solar', 'Wind', 'Ocean', 'Other']
SUBTYPES = {'Biomass': ['w/ CCS', 'w/o CCS'], 'Coal': ['w/ CCS', 'w/o CCS'],
            'Gas': ['w/ CCS', 'w/o CCS'], 'Oil': ['w/ CCS', 'w/o CCS'],
            'Solar': ['PV', 'CSP'], 'Wind': ['Onshore', 'Offshore']}
JAPAN_MODELS = ['AIM/Hub-Japan 2.4', 'AIM/Technology-Japan 2.1',
                'IEEJ-NE_Japan v1', 'TIMES-Japan 3.3 HiCCS']
JAPAN_TECHNOLOGIES = {'coal': 'Coal', 'gas': 'Gas', 'oil': 'Oil',
    'biomass': 'Biomass', 'geothermal': 'Geothermal', 'hydro': 'Hydro',
    'nuclear': 'Nuclear', 'solar_pv': 'Solar|PV', 'ocean': 'Ocean', 'other': 'Other'}


def select_pathways(source_directory, selected):
    """Read the specified model, scenario and region from provider CSV downloads."""
    parts = []
    selected = selected[selected.country.ne('JPN')]
    for filename, group in selected.groupby('source_file'):
        path = Path(source_directory)/filename
        for chunk in pd.read_csv(path, chunksize=50000, low_memory=False):
            chunk.columns = [str(c).strip().title() if not str(c).strip().isdigit()
                             else str(c).strip() for c in chunk.columns]
            use = pd.Series(False, index=chunk.index)
            for row in group.itertuples():
                use |= chunk.Model.eq(row.model) & chunk.Scenario.eq(row.scenario) & chunk.Region.eq(row.source_region)
            use &= chunk.Variable.str.startswith(ELECTRICITY, na=False)
            if use.any():
                parts.append(chunk.loc[use])
    if not parts:
        raise ValueError('The selected electricity pathways are absent from the input files.')
    result = pd.concat(parts, ignore_index=True)
    years = sorted([c for c in result if c.isdigit() and 2020 <= int(c) <= 2050], key=int)
    return result[['Model', 'Scenario', 'Region', 'Variable', 'Unit']+years].drop_duplicates()


def annual_shares(raw, selected, years=np.arange(2022, 2051)):
    """Interpolate generation and avoid counting both technology parents and children."""
    records = []
    for route in selected[selected.country.ne('JPN')].itertuples():
        rows = raw[raw.Model.eq(route.model) & raw.Scenario.eq(route.scenario)
                   & raw.Region.eq(route.source_region)]
        if rows.empty or rows.Variable.duplicated().any():
            raise ValueError('Missing or duplicated electricity variables for '+route.country)
        generation = {}
        for row in rows.to_dict('records'):
            knots = sorted(int(y) for y, value in row.items() if y.isdigit() and pd.notna(value))
            if not knots or knots[0] > min(years) or knots[-1] < max(years):
                raise ValueError('The source years must bracket the requested annual period.')
            scale = {'EJ/yr': 1.0, 'PJ/yr': 0.001}[row['Unit']]
            generation[row['Variable']] = np.interp(years, knots, [row[str(y)]*scale for y in knots])
        for i, year in enumerate(years):
            total = generation[ELECTRICITY][i]
            parents = {t: generation[ELECTRICITY+'|'+t][i] for t in TECHNOLOGIES
                       if ELECTRICITY+'|'+t in generation}
            residual = total-sum(parents.values())
            if total <= 0:
                raise ValueError('Total electricity generation must be positive.')
            if route.country != 'NZL' and residual/total > 0.001:
                parents['Unallocated'] = residual
            denominator = sum(parents.values())
            if denominator <= 0 or min(parents.values()) < 0:
                raise ValueError('Generation components must define a positive nonnegative mix.')
            for technology, amount in parents.items():
                children = {sub: generation[ELECTRICITY+'|'+technology+'|'+sub][i]
                            for sub in SUBTYPES.get(technology, [])
                            if ELECTRICITY+'|'+technology+'|'+sub in generation}
                child_sum = sum(children.values())
                if children and child_sum > 0 and abs(amount-child_sum) <= max(1e-12, total*0.0001):
                    for sub, value in children.items():
                        records.append((route.country, int(year), technology+'|'+sub,
                                        amount/denominator*value/child_sum))
                else:
                    records.append((route.country, int(year), technology, amount/denominator))
    return pd.DataFrame(records, columns=['country', 'year', 'technology', 'share'])


def japan_source_shares(raw, scenario='46by30+100by50'):
    """Extract the four Japanese models at their original reported years.

    Input is the JMIP2_R1_Net_Zero_CDR_database.csv download. Whole unreported
    model-years are omitted. Within reported years, unreported technologies
    are zero, following the study mapping. Wind children replace their parent.
    Shares are normalised over the mapped technology set.
    """
    raw = raw.copy()
    raw.columns = [str(c).lower() for c in raw.columns]
    output = []
    for model in JAPAN_MODELS:
        rows = raw[raw.model.eq(model) & raw.scenario.eq(scenario) & raw.region.eq('Japan')]
        if rows.empty or rows.variable.duplicated().any():
            raise ValueError('Missing or duplicated Japanese source records for '+model)
        by_variable = rows.set_index('variable')
        relevant = by_variable[by_variable.index.str.startswith(ELECTRICITY)]
        for year in range(2020, 2051, 5):
            column = str(year)
            if column not in relevant or not relevant[column].notna().any():
                continue
            def value(suffix):
                variable = ELECTRICITY+'|'+suffix
                if variable not in by_variable.index or pd.isna(by_variable.loc[variable,column]):
                    return 0.0
                row = by_variable.loc[variable]
                return float(row[column])*{'EJ/yr':1.0,'PJ/yr':.001}[row['unit']]
            generation = {key:value(suffix) for key,suffix in JAPAN_TECHNOLOGIES.items()}
            on, off = value('Wind|Onshore'), value('Wind|Offshore')
            if on or off:
                generation.update(wind_onshore=on,wind_offshore=off)
            else:
                generation['wind_unspecified'] = value('Wind')
            total = sum(generation.values())
            if total <= 0 or min(generation.values()) < 0:
                raise ValueError('Invalid Japanese mapped generation.')
            for technology, amount in generation.items():
                output.append((model,year,technology,amount/total))
    return pd.DataFrame(output,columns=['model','year','technology','share'])


def japan_model_factors(shares, technology_factors):
    """Calculate model-specific factors before taking the median.

    technology_factors columns are technology, indicator and factor_per_kwh.
    In the study, Japanese shares are normalised over technologies with matched
    factors. matched_share reports that coverage rather than silently treating
    unmatched technologies as zero impact.
    """
    keys = ['technology','indicator']
    if technology_factors.duplicated(keys).any() or not np.isfinite(technology_factors.factor_per_kwh).all():
        raise ValueError('Japanese technology factors must be finite and unique.')
    matched = shares[shares.technology.isin(technology_factors.technology)].copy()
    coverage = matched.groupby(['model','year'],as_index=False).share.sum().rename(columns={'share':'matched_share'})
    if coverage.matched_share.le(0).any() or len(coverage) != len(shares.groupby(['model','year'])):
        raise ValueError('Every Japanese model-year needs matched technology factors.')
    indicators = technology_factors[['indicator']].drop_duplicates()
    rows = matched.merge(coverage,on=['model','year'],validate='many_to_one').merge(indicators,how='cross')
    rows = rows.merge(technology_factors,on=keys,how='left',validate='many_to_one')
    if rows.factor_per_kwh.isna().any():
        raise ValueError('Every matched technology needs every selected indicator.')
    rows['factor_per_kwh'] *= rows.share/rows.matched_share
    result = rows.groupby(['model','year','indicator'],as_index=False).factor_per_kwh.sum()
    return result.merge(coverage,on=['model','year'],validate='many_to_one')


def japan_annual_median(model_factors, years=np.arange(2022, 2051)):
    """Take indicator medians at reported years, then interpolate as in SI Note 4.

    Input columns are model, year, indicator and factor_per_kwh. These are
    model-specific technology-weighted factors at original reported years,
    not medians of generation shares or already interpolated model factors.
    """
    if model_factors.duplicated(['model', 'year', 'indicator']).any():
        raise ValueError('Each model, source year and indicator must have one factor.')
    if not np.isfinite(model_factors.factor_per_kwh).all():
        raise ValueError('Japanese factors must be finite.')
    medians = model_factors.groupby(['year', 'indicator'], as_index=False).factor_per_kwh.median()
    output = []
    for indicator, rows in medians.groupby('indicator'):
        rows = rows.sort_values('year')
        if rows.year.min() > min(years) or rows.year.max() < max(years):
            raise ValueError('Japanese factor years must bracket the annual period.')
        for year, factor in zip(years, np.interp(years, rows.year, rows.factor_per_kwh)):
            output.append((int(year), indicator, factor))
    return pd.DataFrame(output, columns=['year', 'indicator', 'factor_per_kwh'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True, help='Directory containing the two AR6 v1.1 CSV downloads')
    parser.add_argument('--output', type=Path, default=ROOT/'results/annual_electricity_shares.csv')
    parser.add_argument('--japan-source', type=Path, help='Optional original JMIP2 CSV download')
    parser.add_argument('--japan-technology-factors', type=Path, help='Optional authorised Japanese technology factor CSV')
    args = parser.parse_args()
    selected = pd.read_csv(ROOT/'data/electricity_scenarios.csv')
    result = annual_shares(select_pathways(args.source_dir, selected), selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print('Annual electricity shares saved to '+str(args.output))
    if args.japan_technology_factors and not args.japan_source:
        parser.error('--japan-technology-factors requires --japan-source.')
    if args.japan_source:
        shares = japan_source_shares(pd.read_csv(args.japan_source))
        shares.to_csv(args.output.parent/'japan_source_shares.csv',index=False)
        if args.japan_technology_factors:
            factors = japan_model_factors(shares,pd.read_csv(args.japan_technology_factors))
            factors.to_csv(args.output.parent/'japan_model_factors.csv',index=False)
