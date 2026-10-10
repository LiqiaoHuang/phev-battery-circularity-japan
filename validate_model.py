"""Check annual balances and published base-case physical quantities."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from activities import load_flows
from fleet_model import build_bad_lfp_from_bess_service, LFPParams


def validate(directory, published_base=False):
    tables = load_flows(directory)
    bau = tables['BAU_vehicle_flows'].set_index(['case', 'year'])
    circular = tables['Circular_vehicle_flows'].set_index(['case', 'year'])
    bess = tables['Circular_stationary_storage'].set_index(['case', 'year'])
    lfp = tables['BAU_stationary_storage'].set_index(['case', 'year'])
    for country in ['Japan', 'New Zealand', 'Australia', 'Mongolia', 'Kenya']:
        np.testing.assert_allclose(bau['distance_km_'+country+'_service'],
            circular['distance_km_'+country+'_service'], rtol=1e-12, atol=1e-5)
        if country != 'Japan':
            name = country.replace(' ', '_')
            expected = circular['suppressed_exports_'+name+'_packs'].groupby(level=0).shift(1).fillna(0)
            np.testing.assert_array_equal(circular['altsupply_'+name+'_packs'], expected)
    delivered = bess.LiB_delivered_kWh.groupby(level=0).sum()
    comparator = lfp.lfp_delivered_kWh.groupby(level=0).sum()
    if ((comparator < delivered) | (comparator > delivered*1.0001)).any():
        raise AssertionError('LFP storage must match cumulative delivered service within module rounding.')
    stock = (bess.LiB_started_packs-bess.LiB_retired_packs).groupby(level=0).cumsum()
    if stock.lt(0).any():
        raise AssertionError('Storage retirement cannot exceed additions.')
    zero = build_bad_lfp_from_bess_service(pd.DataFrame({'year': range(2022, 2051),
        'LiB_delivered_kWh': 0.0}), LFPParams())
    if zero.select_dtypes(include='number').drop(columns='year', errors='ignore').to_numpy().any():
        raise AssertionError('Zero service must produce zero LFP flows.')
    if published_base:
        if 'base' not in bau.index.get_level_values('case'):
            raise AssertionError('Published-base checks require the base case.')
        expected = {'LiB_started_packs': 173373, 'LiB_retired_packs': 147598,
                    'LiB_required_input_kWh': 1267296968.6514814,
                    'LiB_delivered_kWh': 970515391.0073903}
        for column, value in expected.items():
            np.testing.assert_allclose(bess.loc['base', column].sum(), value, rtol=1e-10)
        np.testing.assert_equal(lfp.loc['base'].lfp_additions_modules.sum(), 86392)
    print('Annual service, replacement entries, storage balances and zero-service checks passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flows', type=Path, default=Path(__file__).resolve().parent/'results')
    parser.add_argument('--published-base', action='store_true', help='Check the unchanged published base-case quantities.')
    args = parser.parse_args()
    validate(args.flows, args.published_base)
