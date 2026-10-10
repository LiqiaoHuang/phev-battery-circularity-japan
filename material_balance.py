"""Partition recovered-product equivalents in the original manufactured fleet."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from activities import load_flows

LOCATIONS = ['Domestic recovery in Japan', 'Korea and China recovery',
             'In second-life storage at end year', 'Japan landfill',
             'Storage in Mongolia and Kenya', 'In vehicles abroad',
             'In vehicles in Japan at end year']


def export_material_balance(directory):
    tables = load_flows(directory)
    config = tables.get('_parameters') or json.loads((Path(__file__).resolve().parent/'data/parameters.json').read_text())
    mass = sum(config['intervention'][key] for key in ['REC_NISO4_KG', 'REC_COSO4_KG', 'REC_LI2CO3_KG'])
    records = []
    for scenario in ['BAU', 'Circular']:
        for case, flow in tables[scenario+'_vehicle_flows'].groupby('case'):
            exports = flow.filter(regex='^exports_.*_packs$').sum().sum()
            retired = sum(flow['retired_packs_'+country].sum() for country in ['New Zealand', 'Australia', 'Mongolia', 'Kenya'])
            jp_recovered = flow.hydromet_packs_Japan.sum()
            jp_blackmass = flow.bm_packs_Japan_to_Korea.sum()+flow.bm_packs_Japan_to_China.sum()
            abroad_recovered = flow['retired_packs_New Zealand'].sum()+flow['retired_packs_Australia'].sum()
            stored = flow.retired_packs_Mongolia.sum()+flow.retired_packs_Kenya.sum()
            if scenario == 'BAU':
                landfill = flow.disposed_packs_Japan.sum()
                storage_stock = 0
            else:
                landfill = flow.disposed_packs_Japan_outside.sum()
                storage_stock = flow.recovered_to_bess_packs.sum()-flow.bess_retired_packs.sum()
            produced = flow.manufactured_packs.sum()
            jp_vehicles = produced-exports-landfill-jp_blackmass-jp_recovered-storage_stock
            quantities = [jp_recovered, jp_blackmass+abroad_recovered, storage_stock,
                          landfill, stored, exports-retired, jp_vehicles]
            if min(quantities) < 0 or not np.isclose(sum(quantities), produced):
                raise ValueError('The original-fleet material partition does not close.')
            for location, packs in zip(LOCATIONS, quantities):
                records.append(dict(case=case, scenario=scenario, location=location,
                    packs=int(packs), product_mass_kt=float(packs*mass/1e6)))
    result = pd.DataFrame(records)
    result.to_csv(Path(directory)/'material_balance.csv', index=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flows', type=Path, default=Path(__file__).resolve().parent/'results')
    export_material_balance(parser.parse_args().flows)
