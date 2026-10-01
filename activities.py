"""Connect annual fleet outputs to the study's environmental and social factors."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLES = ['BAU_vehicle_flows', 'Circular_vehicle_flows', 'BAU_stationary_storage',
          'Circular_stationary_storage', 'primary_material_makeup']


def load_flows(directory):
    """Read the five annual tables produced by run_model.py."""
    tables = {name: pd.read_csv(Path(directory)/(name+'.csv')) for name in TABLES}
    expected = None
    for name, table in tables.items():
        if not {'case', 'year'} <= set(table) or table.duplicated(['case', 'year']).any():
            raise ValueError('Missing or duplicate case-year keys in '+name)
        keys = set(zip(table['case'], table.year))
        if expected is not None and keys != expected:
            raise ValueError('The annual tables must contain the same case-year keys.')
        expected = keys
    return tables


def build_activities(tables, mapping=None, alternative='A0'):
    """Return signed quantities in the factor reference units.

    The mapping identifies each source table and column. Environmental pack
    production is converted to kg. Social electricity is converted to 1000 kWh.
    Recovered products have negative quantities. Social credit factors use
    their absolute magnitude in impact_calculation.py, matching the study.

    grid_weight is kept separately from quantity because overseas material
    make-up uses recovery timing for the electricity adjustment, not the
    year-by-year make-up mass. Country denotes the foreground activity location,
    not the complete geographical distribution of supply-chain social risk.
    """
    if mapping is None:
        mapping = pd.read_csv(ROOT/'data/activity_mapping.csv').fillna('')
    if mapping.duplicated(['scenario', 'component']).any():
        raise ValueError('Each scenario-component mapping must be unique.')
    results = {'environment': [], 'social': []}
    for rule in mapping.itertuples(index=False):
        source = tables[rule.source_table]
        if rule.source_column not in source:
            raise ValueError('Missing physical column '+rule.source_column)
        raw = pd.to_numeric(source[rule.source_column], errors='raise')
        if not np.isfinite(raw).all() or (raw < 0).any():
            raise ValueError('Physical flows must be finite and nonnegative.')
        row = source[['case', 'year']].copy()
        row['scenario'] = rule.scenario
        row['component'] = rule.component
        row['country'] = rule.country
        row['credit'] = int(rule.credit)
        row['raw_unit'] = rule.raw_unit
        row['raw_quantity'] = raw
        row['electricity_fraction'] = float(rule.electricity_fraction)
        row['grid_weight'] = 0.0
        if rule.grid_weight_column:
            weights = tables[rule.scenario+'_vehicle_flows']
            weights = weights.set_index(['case', 'year'])[rule.grid_weight_column]
            row['grid_weight'] = weights.reindex(pd.MultiIndex.from_frame(row[['case', 'year']])).to_numpy()
            if not np.isfinite(row.grid_weight).all() or row.grid_weight.lt(0).any():
                raise ValueError('Missing or invalid recovery timing weights.')
        sign = -1 if rule.credit else 1
        for assessment in results:
            if assessment == 'environment' and not rule.environmental_included:
                continue
            prefix = 'environmental' if assessment == 'environment' else 'social'
            output = row.copy()
            output['factor_key'] = rule.component if assessment == 'environment' else rule.social_factor_key
            output['activity_unit'] = getattr(rule, prefix+'_unit')
            output['quantity'] = sign*raw*float(getattr(rule, prefix+'_multiplier'))
            results[assessment].append(output)
    results = {key: pd.concat(rows, ignore_index=True) for key, rows in results.items()}
    return replacement_activities(results,tables,alternative)


def replacement_activities(activities, tables, alternative):
    """Apply the A0-A6 battery-only counterfactuals defined in Table S8.

    A0 and A1 have the same incremental manufacturing boundary. LFP capacity
    and retirement are proxies. A4 removes service and is not service equivalent.
    Whole-vehicle production and import transport are outside these adjustments.
    Removed Korean recovery is weighted by replacement-vehicle retirement timing.
    """
    definitions = pd.read_csv(ROOT/'data/replacement_scenarios.csv').set_index('alternative')
    if alternative not in definitions.index:
        raise ValueError('Select A0 to A6 for the replacement assumption.')
    rule = definitions.loc[alternative]
    config = json.loads((ROOT/'data/parameters.json').read_text())
    ratio = config['system']['C_kWh']/config['lfp']['module_kWh']
    circular = tables['Circular_vehicle_flows'].set_index(['case','year'])
    countries = ['New Zealand','Mongolia','Australia','Kenya']
    suppressed = sum(circular['suppressed_exports_'+c.replace(' ','_')+'_packs'] for c in countries)
    retired = sum(circular['retired_packs_'+c+'_altsupply'] for c in countries)
    retired_recycled = circular['retired_packs_New Zealand_altsupply']+circular['retired_packs_Australia_altsupply']
    removed_fraction = rule.new_lfp_fraction+rule.no_replacement_fraction
    deltas = [('Battery_Production_packs',suppressed*rule.new_nmc_fraction),
              ('LFP_Module_Production',suppressed*ratio*rule.new_lfp_fraction),
              ('LFP_Disposal_Japan',retired*ratio*rule.new_lfp_fraction),
              ('Blackmass_Processing_Australia',-retired_recycled*removed_fraction),
              ('Hydromet_Korea_from_Australia_BM',-retired_recycled*removed_fraction)]
    for material, key in [('NiSO4','REC_NISO4_KG'),('CoSO4','REC_COSO4_KG'),('Li2CO3','REC_LI2CO3_KG')]:
        deltas.append(('Avoided_vergin_'+material+'_KR',-retired_recycled*removed_fraction*config['intervention'][key]))
    for country in countries:
        deltas.append(('Electricity_'+country+'_PHEV_service',
                       -circular['required_input_kWh_'+country+'_altsupply']*rule.no_replacement_fraction))
    mapping = pd.read_csv(ROOT/'data/activity_mapping.csv').drop_duplicates('component').set_index('component')
    result = {}
    for assessment, table in activities.items():
        table = table.copy()
        table['activity_role'] = 'baseline'
        parts = [table]
        for component, delta in deltas:
            if not delta.ne(0).any() or (assessment=='environment' and not mapping.loc[component,'environmental_included']):
                continue
            rows = table[table.component.eq(component)]
            scenario = 'Circular' if rows.scenario.eq('Circular').any() else 'BAU'
            rows = rows[rows.scenario.eq(scenario)].copy()
            rows['scenario'] = 'Circular'
            rows['raw_quantity'] = delta.reindex(pd.MultiIndex.from_frame(rows[['case','year']])).to_numpy()
            if rows.electricity_fraction.gt(0).any():
                rows['grid_weight'] = retired_recycled.reindex(pd.MultiIndex.from_frame(rows[['case','year']])).to_numpy()
            prefix = 'environmental' if assessment=='environment' else 'social'
            conversion = float(mapping.loc[component,prefix+'_multiplier'])
            rows['quantity'] = rows.raw_quantity*conversion*np.where(rows.credit.eq(1),-1,1)
            rows['activity_role'] = 'replacement_adjustment'
            parts.append(rows)
        result[assessment] = pd.concat(parts,ignore_index=True).assign(alternative=alternative)
    return result


def export_activities(directory, output=None, alternative='A0'):
    destination = Path(output or directory)
    destination.mkdir(parents=True,exist_ok=True)
    for assessment, table in build_activities(load_flows(directory),alternative=alternative).items():
        table.to_csv(destination/(assessment+'_activities.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flows', type=Path, default=ROOT/'results')
    parser.add_argument('--alternative', choices=['A'+str(i) for i in range(7)],default='A0')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.alternative != 'A0' and (args.output is None or args.output.resolve()==args.flows.resolve()):
        parser.error('Use a separate --output directory for alternative assumptions.')
    export_activities(args.flows,args.output,args.alternative)
