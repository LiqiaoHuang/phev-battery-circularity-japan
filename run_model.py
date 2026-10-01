"""Run the fleet, battery reuse and material recovery calculations."""
from pathlib import Path
import argparse
import json
import warnings

import pandas as pd
import fleet_model as model
from activities import export_activities

ROOT = Path(__file__).resolve().parent


def annual_rows(table):
    """Return annual records without the spreadsheet's cumulative total row."""
    table = table[table.year.astype(str).str.upper().ne('TOTAL')].copy()
    table['year'] = table.year.astype(int)
    return table


def run(case='base', output=ROOT/'results'):
    # Step 1. Load the manufacturing trajectory and model assumptions.
    data = ROOT/'data'
    config = json.loads((data/'parameters.json').read_text())
    manufacture = pd.read_csv(data/'manufacturing_schedule.csv')
    if manufacture.year.duplicated().any() or set(manufacture.year) != set(model.YEARS):
        raise ValueError('Manufacturing data must contain one row for each year from 2022 to 2050.')
    if (manufacture.manufactured_packs < 0).any():
        raise ValueError('Manufactured pack counts cannot be negative.')
    model.MANUFACTURE_COUNTS = dict(zip(manufacture.year.astype(int), manufacture.manufactured_packs.astype(int)))
    model.ROUTE_RATIOS_ORDERED.clear()
    model.ROUTE_RATIOS_ORDERED.update(config['route_ratios'])
    model.SEED_BASE = config['seed_base']
    system = model.SystemParams(**config['system'])
    degradation = model.ModelParams(**config['degradation'])
    lfp_parameters = model.LFPParams(**config['lfp'])
    cases = pd.read_csv(data/'scenarios.csv')
    if case != 'all':
        cases = cases[cases['case'].eq(case)]
    if cases.empty:
        raise ValueError('Select a case listed in data/scenarios.csv or use --case all.')

    # Step 2. Calculate vehicle service, exports and end-of-life flows under BAU.
    print('Calculating BAU vehicle flows', flush=True)
    bau, _, _, _, _ = model.run_bad_scenario_fullfleet(sysP=system, mp=degradation)
    tables = {name: [] for name in ['BAU_vehicle_flows', 'Circular_vehicle_flows',
              'BAU_stationary_storage', 'Circular_stationary_storage', 'primary_material_makeup']}
    for row in cases.to_dict('records'):
        label = row.pop('case')
        parameters = {**config['intervention'], **row}
        print('Calculating '+label, flush=True)

        # Step 3. Allocate captured packs to second-life storage or direct recovery.
        circular, storage, _, _, _ = model.run_bess_scenario_fullfleet(
            sysP=system, mp=degradation, ip=model.InterventionParams(**parameters))

        # Step 4. Size new LFP storage for the same cumulative delivered service.
        lfp = model.build_bad_lfp_from_bess_service(storage, lfp_parameters)

        # Step 5. Calculate primary material make-up for reduced overseas recovery.
        makeup = pd.DataFrame({'year': model.YEARS, **model.compute_virgin_makeup(bau, circular)})
        result = dict(BAU_vehicle_flows=bau, Circular_vehicle_flows=circular,
                      BAU_stationary_storage=lfp, Circular_stationary_storage=storage,
                      primary_material_makeup=makeup)
        for name, table in result.items():
            table = annual_rows(table)
            table.insert(0, 'case', label)
            tables[name].append(table)

    # Step 6. Export annual activity quantities for subsequent impact calculations.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, items in tables.items():
        pd.concat(items, ignore_index=True).to_csv(output/(name+'.csv'), index=False)
    export_activities(output)
    print('Annual flows and impact activity tables saved to '+str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='base', help='base, all, or a case in data/scenarios.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    run(args.case, args.output)
