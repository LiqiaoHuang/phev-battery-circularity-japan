"""Run the fleet, battery reuse and material recovery calculations."""
from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import hashlib
import platform

import pandas as pd
import fleet_model as model
from activities import export_activities
from material_balance import export_material_balance

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
    counts = manufacture.manufactured_packs
    if not np.isfinite(counts).all() or (counts < 0).any() or counts.ne(counts.astype(int)).any():
        raise ValueError('Manufactured pack counts must be finite nonnegative integers.')
    ratios = np.array(list(config['route_ratios'].values()), dtype=float)
    if not np.isfinite(ratios).all() or (ratios < 0).any() or not np.isclose(ratios.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError('Route ratios must be nonnegative and sum to one.')
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
    recovery_yields = tuple(config['intervention'][key] for key in ['REC_NISO4_KG', 'REC_COSO4_KG', 'REC_LI2CO3_KG'])
    bau, _, _, _, _ = model.run_bad_scenario_fullfleet(sysP=system, mp=degradation, recovery_yields=recovery_yields)
    tables = {name: [] for name in ['BAU_vehicle_flows', 'Circular_vehicle_flows',
              'BAU_stationary_storage', 'Circular_stationary_storage', 'primary_material_makeup']}
    for row in cases.to_dict('records'):
        label = row.pop('case')
        parameters = {**config['intervention'], **row}
        rates = [parameters[key] for key in ['export_suppression_rate', 'capture_rate_domestic', 'capture_rate_suppressed_export', 'bess_accept_cap']]
        if not np.isfinite(rates).all() or min(rates) < 0 or max(rates) > 1:
            raise ValueError('Intervention rates must be finite fractions between zero and one.')
        if not model.Y0 <= parameters['recovery_start_year'] <= parameters['bess_operation_start_year'] <= parameters['end_year'] <= model.YN:
            raise ValueError('Intervention years must be ordered within the study horizon.')
        print('Calculating '+label, flush=True)

        # Step 3. Allocate captured packs to second-life storage or direct recovery.
        circular, storage, _, _, _ = model.run_bess_scenario_fullfleet(
            sysP=system, mp=degradation, ip=model.InterventionParams(**parameters))

        # Step 4. Size new LFP storage for the same cumulative delivered service.
        lfp = model.build_bad_lfp_from_bess_service(storage, lfp_parameters,
            start_year=parameters['bess_operation_start_year'], end_year=parameters['end_year'])

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
    metadata = {'parameters': config, 'cases': cases.to_dict('records'),
                'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                'sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(list(ROOT.glob('*.py')) + list(data.iterdir())) if path.is_file()}}
    (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    export_activities(output)
    export_material_balance(output)
    print('Annual flows and impact activity tables saved to '+str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='base', help='base, all, or a case in data/scenarios.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    run(args.case, args.output)
