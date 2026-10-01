"""Calculate impacts from model activities and separately supplied LCIA factors.

Commercial factors are not distributed. Factor CSV columns are factor_key,
activity_unit, indicator and factor. Factors are characterised per reference
unit, before activity multiplication or recycling credit signs are applied.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from electricity import japan_annual_median

ROOT = Path(__file__).resolve().parent


def calculate_impacts(activities, factors, assessment='environment'):
    """Return annual component contributions and cumulative indicator totals.

    Use activities.py outputs. Social factors are already expressed in mrh
    per pack, module, kg or 1000 kWh. Do not multiply them by monetary reference
    amounts again. The 91 retained indicators are selected for the main social
    calculation. Credit factors use absolute magnitudes only for social risk.
    Environmental factors retain their original LCIA signs.
    """
    if assessment not in {'environment', 'social'}:
        raise ValueError('Assessment must be environment or social.')
    keys = ['factor_key', 'activity_unit', 'indicator']
    if not set(keys+['factor']) <= set(factors) or factors.empty:
        raise ValueError('Supply factor_key, activity_unit, indicator and factor.')
    if factors[keys].isna().any().any() or factors.duplicated(keys).any():
        raise ValueError('Factor keys must be present and unique.')
    if not np.isfinite(factors.factor).all() or not np.isfinite(activities.quantity).all():
        raise ValueError('Activities and factors must be finite numeric values.')
    if assessment == 'social':
        indicators = pd.read_csv(ROOT/'data/social_indicators.csv')[['indicator']]
    else:
        indicators = factors[['indicator']].drop_duplicates()
    rows = activities.merge(indicators, how='cross').merge(
        factors[keys+['factor']], on=keys, how='left', validate='many_to_one')
    missing = rows.factor.isna() & rows.quantity.ne(0)
    if missing.any():
        sample = rows.loc[missing, keys].drop_duplicates().head(5).to_dict('records')
        raise ValueError('Missing required factors '+str(sample))
    # A structurally zero activity needs no factor. It is not a missing flow.
    rows.loc[rows.quantity.eq(0) & rows.factor.isna(), 'factor'] = 0.0
    if assessment == 'social':
        rows.loc[rows.credit.eq(1), 'factor'] = rows.loc[rows.credit.eq(1), 'factor'].abs()
    rows['impact'] = rows.quantity*rows.factor
    totals = rows.groupby(['case', 'scenario', 'indicator'], as_index=False).impact.sum()
    return rows, totals


def scenario_changes(totals):
    """Calculate Circular minus BAU as a percentage of BAU for each indicator."""
    result = totals.pivot(index=['case', 'indicator'], columns='scenario', values='impact')
    if not {'BAU', 'Circular'} <= set(result) or result[['BAU', 'Circular']].isna().any().any():
        raise ValueError('Both scenarios are required for each case and indicator.')
    if result.BAU.eq(0).any():
        raise ValueError('Percentage change is undefined for a zero BAU total.')
    result['change_pct'] = 100*(result.Circular/result.BAU-1)
    return result.reset_index()


def electricity_indices(shares, technology_factors):
    """Build indicator-specific foreign electricity indices relative to 2022.

    shares columns are country (ISO3), year, technology and share.
    technology_factors columns are country, technology, indicator and
    factor_per_kwh. Multiple providers must first be weighted by their assigned
    shares within each technology. These are generation factors, not purchased
    electricity baseline factors.
    """
    keys = ['country', 'technology', 'indicator']
    if technology_factors.duplicated(keys).any() or shares.duplicated(keys[:2]+['year']).any():
        raise ValueError('Duplicate technology factors or generation shares.')
    if not np.isfinite(shares.share).all() or shares.share.lt(0).any():
        raise ValueError('Generation shares must be finite and nonnegative.')
    if not np.isfinite(technology_factors.factor_per_kwh).all():
        raise ValueError('Technology factors must be finite.')
    if not np.allclose(shares.groupby(['country', 'year']).share.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError('Generation shares must sum to one.')
    indicators = technology_factors[['indicator']].drop_duplicates()
    rows = shares.merge(indicators, how='cross').merge(
        technology_factors, on=keys, how='left', validate='many_to_one')
    if (rows.factor_per_kwh.isna() & rows.share.ne(0)).any():
        raise ValueError('A positive generation share is missing its technology factor.')
    rows['weighted_factor'] = rows.share*rows.factor_per_kwh.fillna(0)
    result = rows.groupby(['country', 'year', 'indicator'], as_index=False).weighted_factor.sum()
    anchor = result[result.year.eq(2022)][['country', 'indicator', 'weighted_factor']].rename(columns={'weighted_factor': 'factor_2022'})
    result = result.merge(anchor, on=['country', 'indicator'], how='left', validate='many_to_one')
    if result.factor_2022.isna().any() or result.factor_2022.le(0).any():
        raise ValueError('Every country and indicator needs a positive 2022 factor.')
    result['index_vs_2022'] = result.weighted_factor/result.factor_2022
    return result


def apply_environment_grid(rows, foreign_indices, japan_factors):
    """Apply Supplementary Note 4 without changing social factors.

    Foreign explicit electricity uses purchased-electricity baseline factors
    times the generation-mix index. Japan uses absolute annual median factors.
    Embedded CN/KR impacts use the recovery-weighted index and the fractions
    in activity_mapping.csv. Weighting is performed separately by component,
    case, scenario and indicator. Fractions are scenario assumptions.

    foreign_indices columns are country, year, indicator, index_vs_2022.
    japan_factors columns are year, indicator, factor_per_kwh.
    """
    result = rows.copy().reset_index(drop=True)
    result['static_impact'] = result.impact
    keys = ['country', 'year', 'indicator']
    if foreign_indices.duplicated(keys).any() or japan_factors.duplicated(keys[1:]).any():
        raise ValueError('Grid factors must be unique for each country-year-indicator.')
    if not np.isfinite(foreign_indices.index_vs_2022).all() or not np.isfinite(japan_factors.factor_per_kwh).all():
        raise ValueError('Grid values must be finite.')
    grid = foreign_indices[keys+['index_vs_2022']]
    result = result.merge(grid, on=keys, how='left', validate='many_to_one')
    explicit = result.component.str.startswith('Electricity_')
    jp = explicit & result.country.eq('JPN')
    foreign = explicit & ~result.country.eq('JPN')
    embedded = result.electricity_fraction.gt(0)
    need_grid = (foreign | embedded) & (result.quantity.ne(0) | result.grid_weight.ne(0))
    if result.loc[need_grid, 'index_vs_2022'].isna().any():
        raise ValueError('A required foreign country-year-indicator index is missing.')
    result.loc[foreign, 'impact'] *= result.loc[foreign, 'index_vs_2022'].fillna(1)
    japan = result.loc[jp, ['year', 'indicator']].merge(
        japan_factors, on=['year', 'indicator'], how='left', validate='many_to_one')
    if japan.factor_per_kwh.isna().any():
        raise ValueError('A required Japanese annual absolute factor is missing.')
    result.loc[jp, 'impact'] = result.loc[jp, 'quantity'].to_numpy()*japan.factor_per_kwh.to_numpy()
    if embedded.any():
        group = ['case', 'scenario', 'component', 'indicator', 'activity_role']
        w = result.loc[embedded, group+['grid_weight', 'index_vs_2022']].copy()
        w['weighted_index'] = w.grid_weight*w.index_vs_2022.fillna(1)
        numerator = w.groupby(group).weighted_index.transform('sum')
        denominator = w.groupby(group).grid_weight.transform('sum')
        average = (numerator/denominator.replace(0, np.nan)).fillna(1)
        result.loc[embedded, 'impact'] *= 1+result.loc[embedded, 'electricity_fraction']*(average-1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assessment', choices=['environment', 'social'], required=True)
    parser.add_argument('--flows', type=Path, default=ROOT/'results')
    parser.add_argument('--factors', type=Path, required=True)
    parser.add_argument('--grid-shares', type=Path)
    parser.add_argument('--technology-factors', type=Path)
    parser.add_argument('--japan-model-factors', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    grid_args = [args.grid_shares, args.technology_factors, args.japan_model_factors]
    if any(grid_args) and (not all(grid_args) or args.assessment != 'environment'):
        parser.error('Grid adjustment requires all three grid options and the environmental assessment.')
    activities = pd.read_csv(args.flows/(args.assessment+'_activities.csv'))
    rows, _ = calculate_impacts(activities, pd.read_csv(args.factors), args.assessment)
    mode = 'static'
    if all(grid_args):
        indices = electricity_indices(pd.read_csv(args.grid_shares), pd.read_csv(args.technology_factors))
        japan = japan_annual_median(pd.read_csv(args.japan_model_factors))
        rows = apply_environment_grid(rows, indices, japan)
        mode = 'prospective'
    output = args.output or args.flows/(args.assessment+'_'+mode)
    output.mkdir(parents=True, exist_ok=True)
    totals = rows.groupby(['case', 'scenario', 'indicator'], as_index=False).impact.sum()
    rows.groupby(['case', 'scenario', 'component', 'indicator'], as_index=False).impact.sum().to_csv(output/'components.csv', index=False)
    scenario_changes(totals).to_csv(output/'scenario_comparison.csv', index=False)
    if args.assessment == 'social':
        summed = totals.groupby(['case', 'scenario'], as_index=False).impact.sum()
        summed['indicator'] = 'Sum of 91 retained indicators (mrh)'
        scenario_changes(summed).to_csv(output/'social_total.csv', index=False)
    print('Impact results saved to '+str(output))


if __name__ == '__main__':
    main()
