"""Dynamic PHEV fleet and battery flow model for 2022–2050.

The calculation follows Supplementary Notes 1 and 2. Shared degradation,
BAU, Domestic Circular and stationary-service calculations are grouped below.
Use run_model.py to load the study inputs and export annual flow tables.
"""

import math
import datetime
from dataclasses import dataclass
from typing import List
from collections import OrderedDict
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from typing import Dict

# Shared parameters and degradation calculations

YEARS = np.arange(2022, 2051, dtype=int)
Y0, YN = (int(YEARS[0]), int(YEARS[-1]))
nY = len(YEARS)

def jst_now_str():
    tz = datetime.timezone(datetime.timedelta(hours=9))
    return datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S JST')
ROUTE_RATIOS_ORDERED = OrderedDict([('Export_NewZealand', 0.1), ('Export_Mongolia', 0.04), ('Export_Australia', 0.03), ('Export_Kenya', 0.01), ('Domestic_BM_JP', 0.25), ('Domestic_Disposal_JP', 0.57)])
EXPORT_DESTS = ['New Zealand', 'Mongolia', 'Australia', 'Kenya']
DEST_SEED = {'New Zealand': 11, 'Mongolia': 12, 'Australia': 13, 'Kenya': 14}
SEED_BASE = 910000

def year_seed(y: int) -> int:
    return SEED_BASE + int(y) * 13

def seed_dom_params(y: int, is_bm: bool) -> int:
    return year_seed(y) + (101 if is_bm else 102)

def seed_dom_years(y: int, is_bm: bool) -> int:
    return year_seed(y) + (201 if is_bm else 202)

def seed_exp_params(y: int, dest: str) -> int:
    return year_seed(y) + 301 + DEST_SEED[dest]

def seed_exp_years(y: int, dest: str) -> int:
    return year_seed(y) + 401 + DEST_SEED[dest]

@dataclass
class SystemParams:
    C_kWh: float = 18.1
    DoD: float = 0.95
    km_per_kWh: float = 10.3

@dataclass
class ModelParams:
    U_beta_a: float = 2.0
    U_beta_b: float = 2.0
    lambda_knee: float = 1.5
    sigma_knee: float = 0.1
    T50_mean: float = 1079.0
    beta_T: float = -0.5
    sigma_T50: float = 0.1
    phi_const: float = 941.0 / 138.0

def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit(p: float) -> float:
    return math.log(p / (1.0 - p))

def soh_piecewise(t, T_knee, S_knee, dS1, dS2):
    s1 = 1.0 - dS1 * t
    s2 = S_knee - dS2 * (t - T_knee)
    return np.where(t <= T_knee, s1, s2)

def cycles_to_soh_vec(S_target, T_knee, S_knee, dS1, dS2):
    S_target = float(S_target)
    pre = S_target >= S_knee
    t = np.empty_like(T_knee, dtype=np.int32)
    t_pre = np.ceil((1.0 - S_target) / dS1).astype(np.int32)
    t_post = T_knee + np.ceil((S_knee - S_target) / dS2).astype(np.int32)
    t[pre] = t_pre[pre]
    t[~pre] = t_post[~pre]
    return np.maximum(1, t)

def cycles_to_soh_target_vec(S_target_vec, T_knee, S_knee, dS1, dS2):
    S_target_vec = np.asarray(S_target_vec, dtype=float)
    T_knee = np.asarray(T_knee)
    S_knee = np.asarray(S_knee)
    dS1 = np.asarray(dS1)
    dS2 = np.asarray(dS2)
    if S_target_vec.shape != T_knee.shape:
        try:
            S_target_vec = np.broadcast_to(S_target_vec, T_knee.shape)
        except Exception as e:
            raise ValueError(f'S_target_vec shape {S_target_vec.shape} is not compatible with T_knee shape {T_knee.shape}') from e
    pre = S_target_vec >= S_knee
    t = np.empty_like(T_knee, dtype=np.int32)
    t_pre = np.ceil((1.0 - S_target_vec) / dS1).astype(np.int32)
    t_post = T_knee + np.ceil((S_knee - S_target_vec) / dS2).astype(np.int32)
    t[pre] = t_pre[pre]
    t[~pre] = t_post[~pre]
    return np.maximum(1, t)

def gen_params_batch(n: int, seed: int, mp: ModelParams):
    """Sample severity, knee location and two-segment capacity fade for a manufacturing cohort."""
    rg = rng(seed)
    U = rg.beta(mp.U_beta_a, mp.U_beta_b, size=n)
    center = logit(0.7)
    z = center + mp.lambda_knee * (U - 0.5) + rg.normal(0.0, mp.sigma_knee, size=n)
    S_knee = logistic(z)
    Z_T50 = rg.lognormal(mean=-0.5 * mp.sigma_T50 ** 2, sigma=mp.sigma_T50, size=n)
    T50_f = mp.T50_mean * np.exp(mp.beta_T * (U - 0.5)) * Z_T50
    T50 = np.rint(np.maximum(2.0, T50_f)).astype(np.int32)
    phi = np.full(n, mp.phi_const, dtype=float)
    num = phi * (1.0 - S_knee) * T50
    den = S_knee - 0.5 + phi * (1.0 - S_knee)
    T_knee_f = num / den
    T_knee = np.rint(np.clip(T_knee_f, 1, T50 - 1)).astype(np.int32)
    dS1 = (1.0 - S_knee) / T_knee
    dS2 = (S_knee - 0.5) / (T50 - T_knee)
    T80 = cycles_to_soh_vec(0.8, T_knee, S_knee, dS1, dS2)
    a = (0.8 - 0.95) / T80.astype(float)
    return dict(U=U, S_knee=S_knee, T50=T50, T_knee=T_knee, dS1=dS1, dS2=dS2, T80=T80, a=a)

def sum_1_to_T(T):
    return T * (T + 1) / 2.0

def sumsq_1_to_T(T):
    return T * (T + 1) * (2 * T + 1) / 6.0

def sums_piecewise_closed_vec(T, T_knee, S_knee, dS1, dS2, a):
    T = T.astype(np.int64)
    Tk = T_knee.astype(np.int64)
    T1 = np.minimum(T, Tk)
    s_t1 = sum_1_to_T(T1)
    s_t1_2 = sumsq_1_to_T(T1)
    sum_S1 = T1 - dS1 * s_t1
    sum_Seta1 = 0.95 * T1 + (a - 0.95 * dS1) * s_t1 - a * dS1 * s_t1_2
    has2 = T > Tk
    sum_S2 = np.zeros_like(sum_S1, dtype=float)
    sum_Seta2 = np.zeros_like(sum_Seta1, dtype=float)
    if np.any(has2):
        n2 = (T - Tk).astype(np.int64)
        s_t2 = sum_1_to_T(T) - sum_1_to_T(Tk)
        s_t2_2 = sumsq_1_to_T(T) - sumsq_1_to_T(Tk)
        b0 = S_knee + dS2 * Tk
        sum_S2[has2] = b0[has2] * n2[has2] - dS2[has2] * s_t2[has2]
        sum_Seta2[has2] = a[has2] * b0[has2] * s_t2[has2] + 0.95 * b0[has2] * n2[has2] - a[has2] * dS2[has2] * s_t2_2[has2] - 0.95 * dS2[has2] * s_t2[has2]
    return ((sum_S1 + sum_S2).astype(float), (sum_Seta1 + sum_Seta2).astype(float))

def energy_between_cycles_vec(C_kWh, DoD, a, T1, T2, T_knee, S_knee, dS1, dS2):
    """Integrate charging input and delivered energy over the specified cycle interval."""
    T1 = T1.astype(np.int64)
    T2 = T2.astype(np.int64)
    valid = T2 >= T1
    Ein = np.zeros_like(a, dtype=float)
    Eout = np.zeros_like(a, dtype=float)
    if np.any(valid):
        sS2, sSe2 = sums_piecewise_closed_vec(T2[valid], T_knee[valid], S_knee[valid], dS1[valid], dS2[valid], a[valid])
        T0 = (T1[valid] - 1).astype(np.int64)
        sS1, sSe1 = sums_piecewise_closed_vec(T0, T_knee[valid], S_knee[valid], dS1[valid], dS2[valid], a[valid])
        sumS = sS2 - sS1
        sumSe = sSe2 - sSe1
        Ein[valid] = C_kWh * DoD * sumS
        Eout[valid] = C_kWh * DoD * sumSe
    return (Ein, Eout)

def tri_int(rg: np.random.Generator, left: int, mode: int, right: int, size: int):
    x = rg.triangular(left, mode, right, size=size)
    y = np.rint(x).astype(np.int32)
    return np.clip(y, left, right)

def allocate_integer(total: int, ratios: dict):
    keys = list(ratios.keys())
    raw = np.array([ratios[k] * total for k in keys], dtype=float)
    base = np.floor(raw).astype(int)
    rem = raw - base
    leftover = total - base.sum()
    if leftover > 0:
        order = np.argsort(-rem)
        for i in range(leftover):
            base[order[i]] += 1
    return {k: int(v) for k, v in zip(keys, base)}

def add_interval(diff_arr, start_year, end_year, value_per_year):
    s = np.maximum(start_year, Y0)
    e = np.minimum(end_year, YN)
    valid = e >= s
    if not np.any(valid):
        return
    s_idx = (s[valid] - Y0).astype(np.int32)
    e1_idx = (e[valid] - Y0 + 1).astype(np.int32)
    np.add.at(diff_arr, s_idx, value_per_year[valid])
    np.add.at(diff_arr, e1_idx, -value_per_year[valid])

def add_interval_count(diff_arr, start_year, end_year):
    s = np.maximum(start_year, Y0)
    e = np.minimum(end_year, YN)
    valid = e >= s
    if not np.any(valid):
        return
    s_idx = (s[valid] - Y0).astype(np.int32)
    e1_idx = (e[valid] - Y0 + 1).astype(np.int32)
    np.add.at(diff_arr, s_idx, 1)
    np.add.at(diff_arr, e1_idx, -1)

def add_point(count_arr, year, value=1):
    valid = (year >= Y0) & (year <= YN)
    if not np.any(valid):
        return
    idx = (year[valid] - Y0).astype(np.int32)
    if np.isscalar(value):
        np.add.at(count_arr, idx, value)
    else:
        np.add.at(count_arr, idx, value[valid])

def process_stage_const_annual(diff_active_country, diff_in_country, diff_deliv_country, diff_km_country, start_year: np.ndarray, duration_years: np.ndarray, per_in: np.ndarray, per_out: np.ndarray, per_km: np.ndarray):
    """Allocate vehicle service and electricity across each service interval."""
    end_year = start_year + duration_years - 1
    add_interval_count(diff_active_country, start_year, end_year)
    add_interval(diff_in_country, start_year, end_year, per_in)
    add_interval(diff_deliv_country, start_year, end_year, per_out)
    add_interval(diff_km_country, start_year, end_year, per_km)
    return end_year
MANUFACTURE_COUNTS = {2022: 9000, 2023: 24000, 2024: 20000, 2025: 24300, 2026: 28600, 2027: 32900, 2028: 37200, 2029: 41500, 2030: 46000, 2031: 50300, 2032: 54600, 2033: 58900, 2034: 63200, 2035: 67500, 2036: 67500, 2037: 67500, 2038: 67500, 2039: 67500, 2040: 67500, 2041: 67500, 2042: 67500, 2043: 67500, 2044: 67500, 2045: 67500, 2046: 67500, 2047: 67500, 2048: 67500, 2049: 67500, 2050: 67500}
assert set(MANUFACTURE_COUNTS.keys()) == set(YEARS.tolist())

@dataclass
class LFPParams:
    module_kWh: float = 2.4
    DoD: float = 0.968
    eta: float = 0.95
    dSoH_per_cycle: float = 0.04397 / 1000.0
    cycles_per_day: float = 1.0

def _append_total_row(df: pd.DataFrame, year_col: str='year', label: str='TOTAL'):
    if df is None or df.empty:
        return df
    out = df.copy()
    total = {}
    for c in out.columns:
        if c == year_col:
            total[c] = label
        elif np.issubdtype(out[c].dtype, np.number):
            total[c] = float(out[c].sum())
        else:
            total[c] = None
    out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)
    return out


# BAU fleet and end-of-life routes

def run_bad_scenario_fullfleet(seed: int=20251218, sysP=None, mp=None):
    """Calculate BAU vehicle service, exports, retirement and recovery by year."""
    if sysP is None:
        sysP = SystemParams()
    if mp is None:
        mp = ModelParams()
    manufactured = np.array([MANUFACTURE_COUNTS[int(y)] for y in YEARS], dtype=np.int64)
    FULL_N = int(manufactured.sum())
    route_ratios = ROUTE_RATIOS_ORDERED
    S_DOM_LEFT, S_DOM_MODE, S_DOM_RIGHT = (0.7, 0.75, 0.8)
    J_DOM_LEFT, J_DOM_MODE, J_DOM_RIGHT = (10, 15, 20)
    S_EXPORT_LEFT, S_EXPORT_MODE, S_EXPORT_RIGHT = (0.87, 0.9, 0.93)
    J_EXPORT_LEFT, J_EXPORT_MODE, J_EXPORT_RIGHT = (3, 5, 7)
    S_OV_LEFT, S_OV_MODE, S_OV_RIGHT = (0.6, 0.65, 0.7)
    OV_LEFT, OV_MODE, OV_RIGHT = (8, 12, 15)
    REC_NISO4_KG = 24.8
    REC_COSO4_KG = 3.07
    REC_LI2CO3_KG = 7.4
    countries = ['Japan', 'New Zealand', 'Australia', 'Mongolia', 'Kenya']
    c2i = {c: i for i, c in enumerate(countries)}
    nC = len(countries)
    diff_active = np.zeros((nC, nY + 1), dtype=np.int64)
    diff_deliv = np.zeros((nC, nY + 1), dtype=float)
    diff_in = np.zeros((nC, nY + 1), dtype=float)
    diff_km = np.zeros((nC, nY + 1), dtype=float)
    retired = np.zeros((nC, nY), dtype=np.int64)
    bm_jp = np.zeros(nY, dtype=np.int64)
    disp_jp = np.zeros(nY, dtype=np.int64)
    bm_au = np.zeros(nY, dtype=np.int64)
    lfp_additions = np.zeros(nY, dtype=np.int64)
    lfp_active = np.zeros(nY, dtype=np.int64)
    lfp_eout = np.zeros(nY, dtype=float)
    lfp_ein = np.zeros(nY, dtype=float)
    lfp_retired = np.zeros(nY, dtype=np.int64)
    lfp_disposed = np.zeros(nY, dtype=np.int64)
    rg_global = rng(seed)
    sample_size = 1000
    sample_ids_arr = rg_global.choice(FULL_N, size=sample_size, replace=False).astype(np.int64)
    sample_chunks = []
    pid_base = 0
    for y in YEARS:
        y = int(y)
        n_y = MANUFACTURE_COUNTS[y]
        alloc = allocate_integer(n_y, route_ratios)

        def simulate_domestic(n_dom: int, is_bm: bool):
            nonlocal pid_base
            if n_dom <= 0:
                return
            params = gen_params_batch(n_dom, seed_dom_params(y, is_bm), mp)
            T_knee = params['T_knee']
            S_knee = params['S_knee']
            dS1 = params['dS1']
            dS2 = params['dS2']
            a = params['a']
            rg_soh = rng(seed_dom_years(y, is_bm) + 1000)
            S_end = rg_soh.triangular(S_DOM_LEFT, S_DOM_MODE, S_DOM_RIGHT, n_dom).astype(float)
            t_end = cycles_to_soh_target_vec(S_end, T_knee, S_knee, dS1, dS2)
            Ein, Eout = energy_between_cycles_vec(sysP.C_kWh, sysP.DoD, a, np.ones(n_dom, dtype=np.int64), t_end.astype(np.int64), T_knee, S_knee, dS1, dS2)
            km_total = Eout * sysP.km_per_kWh
            rg_year = rng(seed_dom_years(y, is_bm))
            years_dom = tri_int(rg_year, J_DOM_LEFT, J_DOM_MODE, J_DOM_RIGHT, n_dom).astype(np.int32)
            per_in = Ein / years_dom
            per_out = Eout / years_dom
            per_km = km_total / years_dom
            start = np.full(n_dom, y, dtype=np.int32)
            end = process_stage_const_annual(diff_active[c2i['Japan']], diff_in[c2i['Japan']], diff_deliv[c2i['Japan']], diff_km[c2i['Japan']], start, years_dom, per_in, per_out, per_km)
            add_point(retired[c2i['Japan']], end, 1)
            if is_bm:
                add_point(bm_jp, end, 1)
            else:
                add_point(disp_jp, end, 1)
            pids0 = pid_base
            pids1 = pid_base + n_dom
            pids = np.arange(pids0, pids1, dtype=np.int64)
            mask = np.isin(pids, sample_ids_arr)
            if np.any(mask):
                idx = np.where(mask)[0]
                df = pd.DataFrame({'pack_id': pids[idx].astype(int), 'manufacture_year': y, 'route': 'Domestic_BM_JP' if is_bm else 'Domestic_Disposal_JP', 'jp_years': years_dom[idx].astype(int), 'ov_years': [None] * len(idx), 'export_year': [None] * len(idx), 'retire_year': end[idx].astype(int), 't_domestic_end_cycles': t_end[idx].astype(int), 't_final_end_cycles': t_end[idx].astype(int), 'SoH_final_target': S_end[idx], 'Japan_Ein_kWh': Ein[idx], 'Japan_Eout_kWh': Eout[idx], 'Japan_km': km_total[idx], 'Overseas_Ein_kWh_physical': np.zeros(len(idx)), 'Overseas_Eout_kWh_physical': np.zeros(len(idx)), 'Overseas_km_physical': np.zeros(len(idx)), 'Overseas_Ein_kWh_altsupply': np.zeros(len(idx)), 'Overseas_Eout_kWh_altsupply': np.zeros(len(idx)), 'Overseas_km_altsupply': np.zeros(len(idx)), 'Total_Ein_kWh_service': Ein[idx], 'Total_Eout_kWh_service': Eout[idx], 'Total_km_service': km_total[idx], 'suppressed_export': 0, 'captured_network': 0})
                sample_chunks.append(df)
            pid_base += n_dom

        def simulate_export(n_exp: int, dest: str):
            nonlocal pid_base
            if n_exp <= 0:
                return
            params = gen_params_batch(n_exp, seed_exp_params(y, dest), mp)
            T_knee = params['T_knee']
            S_knee = params['S_knee']
            dS1 = params['dS1']
            dS2 = params['dS2']
            a = params['a']
            rg_soh = rng(seed_exp_years(y, dest) + 1000)
            S_export = rg_soh.triangular(S_EXPORT_LEFT, S_EXPORT_MODE, S_EXPORT_RIGHT, n_exp).astype(float)
            S_final = rg_soh.triangular(S_OV_LEFT, S_OV_MODE, S_OV_RIGHT, n_exp).astype(float)
            t90 = cycles_to_soh_target_vec(S_export, T_knee, S_knee, dS1, dS2)
            tF = cycles_to_soh_target_vec(S_final, T_knee, S_knee, dS1, dS2)
            rg = rng(seed_exp_years(y, dest))
            years_jp = tri_int(rg, J_EXPORT_LEFT, J_EXPORT_MODE, J_EXPORT_RIGHT, n_exp).astype(np.int32)
            years_ov = tri_int(rg, OV_LEFT, OV_MODE, OV_RIGHT, n_exp).astype(np.int32)
            EinJ, EoutJ = energy_between_cycles_vec(sysP.C_kWh, sysP.DoD, a, np.ones(n_exp, dtype=np.int64), t90.astype(np.int64), T_knee, S_knee, dS1, dS2)
            EinF, EoutF = energy_between_cycles_vec(sysP.C_kWh, sysP.DoD, a, t90.astype(np.int64) + 1, tF.astype(np.int64), T_knee, S_knee, dS1, dS2)
            total_years = (years_jp + years_ov).astype(np.float64)
            total_Ein = EinJ + EinF
            total_Eout = EoutJ + EoutF
            total_km = total_Eout * sysP.km_per_kWh
            per_in = total_Ein / total_years
            per_out = total_Eout / total_years
            per_km = total_km / total_years
            start_jp = np.full(n_exp, y, dtype=np.int32)
            end_jp = process_stage_const_annual(diff_active[c2i['Japan']], diff_in[c2i['Japan']], diff_deliv[c2i['Japan']], diff_km[c2i['Japan']], start_jp, years_jp, per_in, per_out, per_km)
            export_year = end_jp
            start_ov = export_year + 1
            end_ov = process_stage_const_annual(diff_active[c2i[dest]], diff_in[c2i[dest]], diff_deliv[c2i[dest]], diff_km[c2i[dest]], start_ov.astype(np.int32), years_ov, per_in, per_out, per_km)
            add_point(retired[c2i[dest]], end_ov, 1)
            if dest in ['New Zealand', 'Australia']:
                add_point(bm_au, end_ov, 1)
            pids0 = pid_base
            pids1 = pid_base + n_exp
            pids = np.arange(pids0, pids1, dtype=np.int64)
            mask = np.isin(pids, sample_ids_arr)
            if np.any(mask):
                idx = np.where(mask)[0]
                df = pd.DataFrame({'pack_id': pids[idx].astype(int), 'manufacture_year': y, 'route': f"Export_{dest.replace(' ', '_')}", 'jp_years': years_jp[idx].astype(int), 'ov_years': years_ov[idx].astype(int), 'export_year': export_year[idx].astype(int), 'retire_year': end_ov[idx].astype(int), 't_domestic_end_cycles': t90[idx].astype(int), 't_final_end_cycles': tF[idx].astype(int), 'SoH_final_target': S_final[idx], 'Japan_Ein_kWh': EinJ[idx], 'Japan_Eout_kWh': EoutJ[idx], 'Japan_km': EoutJ[idx] * sysP.km_per_kWh, 'Overseas_Ein_kWh_physical': EinF[idx], 'Overseas_Eout_kWh_physical': EoutF[idx], 'Overseas_km_physical': EoutF[idx] * sysP.km_per_kWh, 'Overseas_Ein_kWh_altsupply': np.zeros(len(idx)), 'Overseas_Eout_kWh_altsupply': np.zeros(len(idx)), 'Overseas_km_altsupply': np.zeros(len(idx)), 'Total_Ein_kWh_service': total_Ein[idx], 'Total_Eout_kWh_service': total_Eout[idx], 'Total_km_service': total_km[idx], 'suppressed_export': 0, 'captured_network': 0})
                sample_chunks.append(df)
            pid_base += n_exp
        simulate_export(alloc['Export_NewZealand'], 'New Zealand')
        simulate_export(alloc['Export_Mongolia'], 'Mongolia')
        simulate_export(alloc['Export_Australia'], 'Australia')
        simulate_export(alloc['Export_Kenya'], 'Kenya')
        simulate_domestic(alloc['Domestic_BM_JP'], True)
        simulate_domestic(alloc['Domestic_Disposal_JP'], False)
    assert pid_base == FULL_N, f'pack_id mismatch: {pid_base} != {FULL_N}'
    active = np.cumsum(diff_active[:, :-1], axis=1)
    deliv = np.cumsum(diff_deliv[:, :-1], axis=1)
    ein = np.cumsum(diff_in[:, :-1], axis=1)
    km = np.cumsum(diff_km[:, :-1], axis=1)
    kr_from_jp = bm_jp // 2
    cn_from_jp = bm_jp - kr_from_jp
    hydro_kr = kr_from_jp + bm_au
    hydro_cn = cn_from_jp
    kr_Ni = hydro_kr.astype(float) * REC_NISO4_KG
    kr_Co = hydro_kr.astype(float) * REC_COSO4_KG
    kr_Li = hydro_kr.astype(float) * REC_LI2CO3_KG
    cn_Ni = hydro_cn.astype(float) * REC_NISO4_KG
    cn_Co = hydro_cn.astype(float) * REC_COSO4_KG
    cn_Li = hydro_cn.astype(float) * REC_LI2CO3_KG
    df_annual = pd.DataFrame({'year': YEARS, 'manufactured_packs': manufactured.astype(int)})
    for c in countries:
        i = c2i[c]
        df_annual[f'active_packs_{c}'] = active[i].astype(int)
        df_annual[f'distance_km_{c}'] = km[i]
        df_annual[f'delivered_kWh_{c}'] = deliv[i]
        df_annual[f'required_input_kWh_{c}'] = ein[i]
        df_annual[f'retired_packs_{c}'] = retired[i].astype(int)
        df_annual[f'active_packs_{c}_altsupply'] = 0
        df_annual[f'distance_km_{c}_altsupply'] = 0.0
        df_annual[f'delivered_kWh_{c}_altsupply'] = 0.0
        df_annual[f'required_input_kWh_{c}_altsupply'] = 0.0
        df_annual[f'retired_packs_{c}_altsupply'] = 0
        df_annual[f'active_packs_{c}_service'] = df_annual[f'active_packs_{c}']
        df_annual[f'distance_km_{c}_service'] = df_annual[f'distance_km_{c}']
        df_annual[f'delivered_kWh_{c}_service'] = df_annual[f'delivered_kWh_{c}']
        df_annual[f'required_input_kWh_{c}_service'] = df_annual[f'required_input_kWh_{c}']
        df_annual[f'retired_packs_{c}_service'] = df_annual[f'retired_packs_{c}']
    df_annual['bm_processed_packs_Japan'] = bm_jp.astype(int)
    df_annual['disposed_packs_Japan'] = disp_jp.astype(int)
    df_annual['bm_processed_packs_Australia'] = bm_au.astype(int)
    df_annual['bm_packs_Australia_to_Korea'] = bm_au.astype(int)
    df_annual['bm_packs_Japan_to_Korea'] = kr_from_jp.astype(int)
    df_annual['bm_packs_Japan_to_China'] = cn_from_jp.astype(int)
    df_annual['hydromet_packs_Korea'] = hydro_kr.astype(int)
    df_annual['hydromet_packs_China'] = hydro_cn.astype(int)
    df_annual['NiSO4_kg_Korea'] = kr_Ni
    df_annual['CoSO4_kg_Korea'] = kr_Co
    df_annual['Li2CO3_kg_Korea'] = kr_Li
    df_annual['NiSO4_kg_China'] = cn_Ni
    df_annual['CoSO4_kg_China'] = cn_Co
    df_annual['Li2CO3_kg_China'] = cn_Li
    df_annual['hydromet_packs_Japan'] = 0
    df_annual['NiSO4_kg_Japan'] = 0.0
    df_annual['CoSO4_kg_Japan'] = 0.0
    df_annual['Li2CO3_kg_Japan'] = 0.0
    for _c in ('Korea', 'China', 'Japan'):
        df_annual[f'NiSO4_kg_virgin_makeup_{_c}'] = 0.0
        df_annual[f'CoSO4_kg_virgin_makeup_{_c}'] = 0.0
        df_annual[f'Li2CO3_kg_virgin_makeup_{_c}'] = 0.0
    df_lfp = pd.DataFrame({'year': YEARS, 'lfp_additions_modules': lfp_additions.astype(int), 'lfp_active_modules': lfp_active.astype(int), 'lfp_delivered_kWh': lfp_eout, 'lfp_required_input_kWh': lfp_ein, 'lfp_retired_modules': lfp_retired.astype(int), 'lfp_disposed_modules': lfp_disposed.astype(int)})
    df_sample1000 = pd.concat(sample_chunks, ignore_index=True).sort_values('pack_id') if sample_chunks else pd.DataFrame()
    readme = pd.DataFrame({'note': ['Bad scenario output. (README simplified in split version)']})
    meta = {'FULL_FLEET_total_packs': FULL_N, 'generated_at': jst_now_str(), 'seed': seed}
    return (df_annual, df_lfp, df_sample1000, readme, meta)


# Domestic Circular fleet and stationary storage

@dataclass
class InterventionParams:
    recovery_start_year: int = 2035
    bess_operation_start_year: int = 2035
    export_suppression_rate: float = 0.5
    capture_rate_domestic: float = 0.5
    capture_rate_suppressed_export: float = 1.0
    bess_accept_cap: float = 0.5
    bess_cycles_per_day: float = 1.0
    bess_eol_soh: float = 0.5
    end_year: int = YN
    rng_seed: int = 42
    REC_NISO4_KG: float = 24.8
    REC_COSO4_KG: float = 3.07
    REC_LI2CO3_KG: float = 7.4

def simulate_bess_daily_from_candidates(sysP: SystemParams, mp: ModelParams, ip: InterventionParams, candidates_by_year: Dict[int, np.ndarray], seed: int=0) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int], Optional[int]]:
    """Prioritise remaining cycle life and accumulate daily stationary-storage service."""

    def yidx_local(y: int) -> int:
        return int(y) - int(Y0)

    def days_in_year(y: int) -> int:
        return 366 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 365
    ein_by_year = np.zeros(nY, dtype=float)
    eout_by_year = np.zeros(nY, dtype=float)
    active_pack_days = np.zeros(nY, dtype=float)
    started_packs = np.zeros(nY, dtype=int)
    retired_packs = np.zeros(nY, dtype=int)
    immediate_recycle = np.zeros(nY, dtype=int)
    end_ord = datetime.date(int(ip.end_year), 12, 31).toordinal()
    C = float(sysP.C_kWh)
    DOD = float(sysP.DoD)
    any_bess_used = False
    for capture_year in range(int(ip.recovery_start_year), int(ip.end_year) + 1):
        arr = candidates_by_year.get(capture_year)
        if arr is None or len(arr) == 0:
            continue
        m = int(arr.shape[0])
        cap = float(ip.bess_accept_cap)
        n_accept = int(np.floor(cap * m + 1e-09))
        n_accept = max(0, min(n_accept, m))
        n_immediate = m - n_accept
        immediate_recycle[yidx_local(capture_year)] += int(n_immediate)
        if n_accept <= 0:
            continue
        rng_y = np.random.default_rng(int(seed + capture_year))
        t_entry_all = arr[:, 6].astype(int)
        t_eol_all = cycles_to_soh_vec(float(ip.bess_eol_soh), arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]).astype(int)
        rem_all = np.maximum(t_eol_all - t_entry_all, 0)
        if n_accept == m:
            accept_idx = np.arange(m, dtype=int)
        else:
            tie = rng_y.random(m)
            order = np.lexsort((tie, -rem_all))
            accept_idx = order[:n_accept]
        sub = arr[accept_idx, :]
        Tk = sub[:, 1].astype(float)
        Sk = sub[:, 2].astype(float)
        d1 = sub[:, 3].astype(float)
        d2 = sub[:, 4].astype(float)
        a = sub[:, 5].astype(float)
        t_entry = sub[:, 6].astype(int)
        entry_year = max(int(capture_year), int(ip.bess_operation_start_year))
        if entry_year > int(YN):
            continue
        days_entry = days_in_year(entry_year)
        entry_doy = rng_y.integers(1, days_entry + 1, size=n_accept, dtype=np.int64)
        t_eol = cycles_to_soh_vec(float(ip.bess_eol_soh), Tk, Sk, d1, d2).astype(int)
        rem_total = np.maximum(t_eol - t_entry, 0).astype(int)
        mask_use = rem_total > 0
        n_use = int(np.sum(mask_use))
        n_unusable = int(n_accept - n_use)
        if n_unusable > 0:
            immediate_recycle[yidx_local(capture_year)] += n_unusable
        if n_use <= 0:
            continue
        Tk = Tk[mask_use]
        Sk = Sk[mask_use]
        d1 = d1[mask_use]
        d2 = d2[mask_use]
        a = a[mask_use]
        t_entry_use = t_entry[mask_use]
        entry_doy_use = entry_doy[mask_use]
        rem_total_use = rem_total[mask_use]
        started_packs[yidx_local(entry_year)] += n_use
        any_bess_used = True
        base_ord = datetime.date(entry_year, 1, 1).toordinal()
        s_ord = base_ord + (entry_doy_use - 1)
        e_ord = s_ord + (rem_total_use - 1)
        rem = rem_total_use.copy()
        t_cur = t_entry_use.copy()
        days0 = (days_entry - entry_doy_use + 1).astype(int)
        cyc0 = np.minimum(rem, days0).astype(int)
        mask0 = cyc0 > 0
        if np.any(mask0):
            T1 = t_cur[mask0] + 1
            T2 = t_cur[mask0] + cyc0[mask0]
            ein0, eout0 = energy_between_cycles_vec(C, DOD, a[mask0], T1, T2, Tk[mask0], Sk[mask0], d1[mask0], d2[mask0])
            ein_by_year[yidx_local(entry_year)] += float(np.sum(ein0))
            eout_by_year[yidx_local(entry_year)] += float(np.sum(eout0))
            active_pack_days[yidx_local(entry_year)] += float(np.sum(cyc0[mask0]))
        t_cur += cyc0
        rem -= cyc0
        year_seg = entry_year
        while True:
            ongoing = rem > 0
            if not np.any(ongoing):
                break
            year_seg += 1
            if year_seg > int(YN):
                break
            idx = np.where(ongoing)[0]
            days_y = days_in_year(year_seg)
            cyc = np.minimum(rem[idx], days_y).astype(int)
            if np.any(cyc > 0):
                idx2 = idx[cyc > 0]
                cyc2 = cyc[cyc > 0]
                T1 = t_cur[idx2] + 1
                T2 = t_cur[idx2] + cyc2
                ein_seg, eout_seg = energy_between_cycles_vec(C, DOD, a[idx2], T1, T2, Tk[idx2], Sk[idx2], d1[idx2], d2[idx2])
                ein_by_year[yidx_local(year_seg)] += float(np.sum(ein_seg))
                eout_by_year[yidx_local(year_seg)] += float(np.sum(eout_seg))
                active_pack_days[yidx_local(year_seg)] += float(np.sum(cyc2))
            t_cur[idx] += cyc
            rem[idx] -= cyc
        mask_ret_h = e_ord <= end_ord
        if np.any(mask_ret_h):
            eol_years = pd.to_datetime(e_ord[mask_ret_h] - 719163, unit='D').year.to_numpy()
            for yy in np.unique(eol_years):
                yy_int = int(yy)
                if int(Y0) <= yy_int <= int(YN):
                    retired_packs[yidx_local(yy_int)] += int(np.sum(eol_years == yy))
    rows = []
    for y in range(int(ip.bess_operation_start_year), int(YN) + 1):
        days_y = days_in_year(y)
        rows.append(dict(year=y, LiB_active_packs_avg=active_pack_days[yidx_local(y)] / float(days_y) if days_y > 0 else 0.0, LiB_pack_days=active_pack_days[yidx_local(y)], LiB_required_input_kWh=ein_by_year[yidx_local(y)], LiB_delivered_kWh=eout_by_year[yidx_local(y)], LiB_started_packs=int(started_packs[yidx_local(y)]), LiB_retired_packs=int(retired_packs[yidx_local(y)])))
    df_bess_annual = pd.DataFrame(rows)
    hydromet_from_bess_recycled = {y: int(retired_packs[yidx_local(y)]) for y in range(int(ip.bess_operation_start_year), int(ip.end_year) + 1)}
    hydromet_unused_pool = {y: int(immediate_recycle[yidx_local(y)]) for y in range(int(ip.recovery_start_year), int(ip.end_year) + 1)}
    bess_start_actual = int(ip.bess_operation_start_year) if any_bess_used else None
    return (df_bess_annual, hydromet_from_bess_recycled, hydromet_unused_pool, bess_start_actual)

def run_bess_scenario_fullfleet(seed: int=20251218, ip: Optional[InterventionParams]=None, sysP=None, mp=None):
    """Calculate export suppression, domestic capture, second life and material recovery."""
    if sysP is None:
        sysP = SystemParams()
    if mp is None:
        mp = ModelParams()
    if ip is None:
        ip = InterventionParams()
    manufactured = np.array([MANUFACTURE_COUNTS[int(y)] for y in YEARS], dtype=np.int64)
    FULL_N = int(manufactured.sum())
    route_ratios = ROUTE_RATIOS_ORDERED
    export_dests = EXPORT_DESTS
    S_DOM_LEFT, S_DOM_MODE, S_DOM_RIGHT = (0.7, 0.75, 0.8)
    J_DOM_LEFT, J_DOM_MODE, J_DOM_RIGHT = (10, 15, 20)
    S_EXPORT_LEFT, S_EXPORT_MODE, S_EXPORT_RIGHT = (0.87, 0.9, 0.93)
    J_EXPORT_LEFT, J_EXPORT_MODE, J_EXPORT_RIGHT = (3, 5, 7)
    S_OV_LEFT, S_OV_MODE, S_OV_RIGHT = (0.6, 0.65, 0.7)
    OV_LEFT, OV_MODE, OV_RIGHT = (8, 12, 15)
    countries = ['Japan', 'New Zealand', 'Australia', 'Mongolia', 'Kenya']
    c2i = {c: i for i, c in enumerate(countries)}
    nC = len(countries)
    diff_active = np.zeros((nC, nY + 1), dtype=np.int64)
    diff_deliv = np.zeros((nC, nY + 1), dtype=float)
    diff_in = np.zeros((nC, nY + 1), dtype=float)
    diff_km = np.zeros((nC, nY + 1), dtype=float)
    retired = np.zeros((nC, nY), dtype=np.int64)
    diff_active_alt = np.zeros((nC, nY + 1), dtype=np.int64)
    diff_deliv_alt = np.zeros((nC, nY + 1), dtype=float)
    diff_in_alt = np.zeros((nC, nY + 1), dtype=float)
    diff_km_alt = np.zeros((nC, nY + 1), dtype=float)
    retired_alt = np.zeros((nC, nY), dtype=np.int64)
    bm_jp_outside = np.zeros(nY, dtype=np.int64)
    disp_jp_outside = np.zeros(nY, dtype=np.int64)
    bm_au = np.zeros(nY, dtype=np.int64)
    recovered_network_inflow = np.zeros(nY, dtype=np.int64)
    recovered_from_domestic = np.zeros(nY, dtype=np.int64)
    recovered_from_suppressed = np.zeros(nY, dtype=np.int64)
    recovered_immediate_hydromet = np.zeros(nY, dtype=np.int64)
    suppressed_exports_by_dest = {d: np.zeros(nY, dtype=np.int64) for d in export_dests}
    alt_supply_by_dest = {d: np.zeros(nY, dtype=np.int64) for d in export_dests}
    candidates_by_year: Dict[int, List[np.ndarray]] = {y: [] for y in range(ip.recovery_start_year, ip.end_year + 1)}
    rg_global = rng(seed)
    sample_size = 1000
    sample_ids_arr = rg_global.choice(FULL_N, size=sample_size, replace=False).astype(np.int64)
    sample_chunks = []
    pid_base = 0

    def process_stage_physical(country: str, start_year, duration_years, per_in, per_out, per_km):
        i = c2i[country]
        return process_stage_const_annual(diff_active[i], diff_in[i], diff_deliv[i], diff_km[i], start_year, duration_years, per_in, per_out, per_km)

    def process_stage_alt(country: str, start_year, duration_years, per_in, per_out, per_km):
        i = c2i[country]
        return process_stage_const_annual(diff_active_alt[i], diff_in_alt[i], diff_deliv_alt[i], diff_km_alt[i], start_year, duration_years, per_in, per_out, per_km)

    def push_candidates(event_year: np.ndarray, delta_kWh: np.ndarray, T_knee: np.ndarray, S_knee: np.ndarray, dS1: np.ndarray, dS2: np.ndarray, a: np.ndarray, t_entry: np.ndarray, SoH_entry: np.ndarray):
        for yy in np.unique(event_year):
            yy = int(yy)
            if yy < ip.recovery_start_year or yy > ip.end_year:
                continue
            mask = event_year == yy
            if not np.any(mask):
                continue
            arr = np.vstack([delta_kWh[mask], T_knee[mask].astype(float), S_knee[mask], dS1[mask], dS2[mask], a[mask], t_entry[mask].astype(float), SoH_entry[mask]]).T
            candidates_by_year[yy].append(arr)
    for y in YEARS:
        y = int(y)
        n_y = MANUFACTURE_COUNTS[y]
        yseed = year_seed(y)
        alloc = allocate_integer(n_y, route_ratios)
        alloc_exp = {'New Zealand': alloc['Export_NewZealand'], 'Mongolia': alloc['Export_Mongolia'], 'Australia': alloc['Export_Australia'], 'Kenya': alloc['Export_Kenya']}

        def simulate_domestic_group(n_dom: int, is_bm: bool):
            nonlocal pid_base
            if n_dom <= 0:
                return
            params = gen_params_batch(n_dom, seed_dom_params(y, is_bm), mp)
            T_knee = params['T_knee']
            S_knee = params['S_knee']
            dS1 = params['dS1']
            dS2 = params['dS2']
            a = params['a']
            rg_soh = rng(seed_dom_years(y, is_bm) + 1000)
            S_end = rg_soh.triangular(S_DOM_LEFT, S_DOM_MODE, S_DOM_RIGHT, n_dom).astype(float)
            t_end = cycles_to_soh_target_vec(S_end, T_knee, S_knee, dS1, dS2)
            Ein, Eout = energy_between_cycles_vec(sysP.C_kWh, sysP.DoD, a, np.ones(n_dom, dtype=np.int64), t_end.astype(np.int64), T_knee, S_knee, dS1, dS2)
            km_total = Eout * sysP.km_per_kWh
            rg = rng(seed_dom_years(y, is_bm))
            years_dom = tri_int(rg, J_DOM_LEFT, J_DOM_MODE, J_DOM_RIGHT, n_dom).astype(np.int32)
            per_in = Ein / years_dom
            per_out = Eout / years_dom
            per_km = km_total / years_dom
            start = np.full(n_dom, y, dtype=np.int32)
            end = process_stage_physical('Japan', start, years_dom, per_in, per_out, per_km)
            add_point(retired[c2i['Japan']], end, 1)
            capture_active = end >= ip.recovery_start_year
            cap_flag = np.zeros(n_dom, dtype=bool)
            if np.any(capture_active):
                rg_cap = rng(yseed + 7777 + (1 if is_bm else 2))
                cap_flag[capture_active] = rg_cap.random(np.count_nonzero(capture_active)) < float(ip.capture_rate_domestic)
            outside = ~cap_flag
            if np.any(outside):
                if is_bm:
                    add_point(bm_jp_outside, end[outside], 1)
                else:
                    add_point(disp_jp_outside, end[outside], 1)
            inside = cap_flag
            if np.any(inside):
                add_point(recovered_network_inflow, end[inside].astype(int), 1)
                add_point(recovered_from_domestic, end[inside].astype(int), 1)
                SoH_entry = soh_piecewise(t_end[inside].astype(np.int64), T_knee[inside].astype(np.int64), S_knee[inside].astype(float), dS1[inside].astype(float), dS2[inside].astype(float)).astype(float)
                delta_kWh = sysP.C_kWh * SoH_entry
                push_candidates(event_year=end[inside].astype(int), delta_kWh=delta_kWh.astype(float), T_knee=T_knee[inside], S_knee=S_knee[inside], dS1=dS1[inside], dS2=dS2[inside], a=a[inside], t_entry=t_end[inside].astype(int), SoH_entry=SoH_entry.astype(float))
            pids0 = pid_base
            pids1 = pid_base + n_dom
            pids = np.arange(pids0, pids1, dtype=np.int64)
            mask = np.isin(pids, sample_ids_arr)
            if np.any(mask):
                idx = np.where(mask)[0]
                df = pd.DataFrame({'pack_id': pids[idx].astype(int), 'manufacture_year': y, 'route': 'Domestic_BM_JP' if is_bm else 'Domestic_Disposal_JP', 'jp_years': years_dom[idx].astype(int), 'ov_years': [None] * len(idx), 'export_year': [None] * len(idx), 'retire_year': end[idx].astype(int), 'captured_network': cap_flag[idx].astype(int), 'suppressed_export': 0, 'Japan_Ein_kWh': Ein[idx], 'Japan_Eout_kWh': Eout[idx], 'Japan_km': km_total[idx], 'Overseas_Ein_kWh_physical': np.zeros(len(idx)), 'Overseas_Eout_kWh_physical': np.zeros(len(idx)), 'Overseas_km_physical': np.zeros(len(idx)), 'Overseas_Ein_kWh_altsupply': np.zeros(len(idx)), 'Overseas_Eout_kWh_altsupply': np.zeros(len(idx)), 'Overseas_km_altsupply': np.zeros(len(idx)), 'Total_Ein_kWh_service': Ein[idx], 'Total_Eout_kWh_service': Eout[idx], 'Total_km_service': km_total[idx]})
                sample_chunks.append(df)
            pid_base += n_dom

        def simulate_export_group(n_exp: int, dest: str):
            nonlocal pid_base
            if n_exp <= 0:
                return
            params = gen_params_batch(n_exp, seed_exp_params(y, dest), mp)
            T_knee = params['T_knee']
            S_knee = params['S_knee']
            dS1 = params['dS1']
            dS2 = params['dS2']
            a = params['a']
            rg_soh = rng(seed_exp_years(y, dest) + 1000)
            S_export = rg_soh.triangular(S_EXPORT_LEFT, S_EXPORT_MODE, S_EXPORT_RIGHT, n_exp).astype(float)
            S_final = rg_soh.triangular(S_OV_LEFT, S_OV_MODE, S_OV_RIGHT, n_exp).astype(float)
            t90 = cycles_to_soh_target_vec(S_export, T_knee, S_knee, dS1, dS2)
            tF = cycles_to_soh_target_vec(S_final, T_knee, S_knee, dS1, dS2)
            rg = rng(seed_exp_years(y, dest))
            years_jp = tri_int(rg, J_EXPORT_LEFT, J_EXPORT_MODE, J_EXPORT_RIGHT, n_exp).astype(np.int32)
            years_ov = tri_int(rg, OV_LEFT, OV_MODE, OV_RIGHT, n_exp).astype(np.int32)
            start_jp = np.full(n_exp, y, dtype=np.int32)
            EinJ, EoutJ = energy_between_cycles_vec(sysP.C_kWh, sysP.DoD, a, np.ones(n_exp, dtype=np.int64), t90.astype(np.int64), T_knee, S_knee, dS1, dS2)
            EinF, EoutF = energy_between_cycles_vec(sysP.C_kWh, sysP.DoD, a, t90.astype(np.int64) + 1, tF.astype(np.int64), T_knee, S_knee, dS1, dS2)
            end_jp_tmp = start_jp + years_jp - 1
            export_year = end_jp_tmp
            start_ov = export_year + 1
            sup_eligible = export_year >= ip.recovery_start_year
            sup_flag = np.zeros(n_exp, dtype=bool)
            if np.any(sup_eligible):
                rg_sup = rng(yseed + 6000 + DEST_SEED[dest])
                sup_flag[sup_eligible] = rg_sup.random(np.count_nonzero(sup_eligible)) < float(ip.export_suppression_rate)
            keep = ~sup_flag
            end_keep = np.full(n_exp, -1, dtype=np.int32)
            if np.any(keep):
                tot_years = (years_jp[keep] + years_ov[keep]).astype(np.float64)
                tot_Ein = EinJ[keep] + EinF[keep]
                tot_Eout = EoutJ[keep] + EoutF[keep]
                tot_km = tot_Eout * sysP.km_per_kWh
                per_in = tot_Ein / tot_years
                per_out = tot_Eout / tot_years
                per_km = tot_km / tot_years
                end_jp = process_stage_physical('Japan', start_jp[keep], years_jp[keep], per_in, per_out, per_km)
                end_keep[keep] = process_stage_physical(dest, start_ov[keep].astype(np.int32), years_ov[keep], per_in, per_out, per_km).astype(np.int32)
                add_point(retired[c2i[dest]], end_keep[keep], 1)
                if dest in ['New Zealand', 'Australia']:
                    add_point(bm_au, end_keep[keep], 1)
            if np.any(sup_flag):
                add_point(suppressed_exports_by_dest[dest], export_year[sup_flag], 1)
                n_sup = int(sup_flag.sum())
                tot_years_base = years_jp[sup_flag] + years_ov[sup_flag]
                tot_Ein_base = EinJ[sup_flag] + EinF[sup_flag]
                tot_Eout_base = EoutJ[sup_flag] + EoutF[sup_flag]
                per_in_base = tot_Ein_base / tot_years_base
                per_out_base = tot_Eout_base / tot_years_base
                per_km_base = per_out_base * sysP.km_per_kWh
                end_alt = process_stage_alt(dest, start_ov[sup_flag], years_ov[sup_flag], per_in_base, per_out_base, per_km_base)
                add_point(retired_alt[c2i[dest]], end_alt, 1)
                if dest in ['New Zealand', 'Australia']:
                    add_point(bm_au, end_alt, 1)
                _ = process_stage_physical('Japan', start_jp[sup_flag], years_jp[sup_flag], per_in_base, per_out_base, per_km_base)
                add_point(retired[c2i['Japan']], export_year[sup_flag], 1)
                if float(ip.capture_rate_suppressed_export) >= 1.0:
                    cap2 = np.ones(n_sup, dtype=bool)
                else:
                    rg_cap = rng(yseed + 9000 + DEST_SEED[dest])
                    cap2 = rg_cap.random(n_sup) < float(ip.capture_rate_suppressed_export)
                if np.any(cap2):
                    ev_year = export_year[sup_flag][cap2].astype(int)
                    add_point(recovered_network_inflow, ev_year, 1)
                    add_point(recovered_from_suppressed, ev_year, 1)
                    SoH_entry = soh_piecewise(t90[sup_flag][cap2], T_knee[sup_flag][cap2], S_knee[sup_flag][cap2], dS1[sup_flag][cap2], dS2[sup_flag][cap2])
                    delta_kWh = sysP.C_kWh * SoH_entry
                    push_candidates(event_year=ev_year, delta_kWh=delta_kWh.astype(float), T_knee=T_knee[sup_flag][cap2], S_knee=S_knee[sup_flag][cap2], dS1=dS1[sup_flag][cap2], dS2=dS2[sup_flag][cap2], a=a[sup_flag][cap2], t_entry=t90[sup_flag][cap2].astype(int), SoH_entry=SoH_entry.astype(float))
            pids0 = pid_base
            pids1 = pid_base + n_exp
            pids = np.arange(pids0, pids1, dtype=np.int64)
            mask = np.isin(pids, sample_ids_arr)
            if np.any(mask):
                idx = np.where(mask)[0]
                EinF_phys = np.zeros(n_exp, dtype=float)
                EoutF_phys = np.zeros(n_exp, dtype=float)
                kmF_phys = np.zeros(n_exp, dtype=float)
                EinF_alt = np.zeros(n_exp, dtype=float)
                EoutF_alt = np.zeros(n_exp, dtype=float)
                kmF_alt = np.zeros(n_exp, dtype=float)
                EinF_phys[keep] = EinF[keep]
                EoutF_phys[keep] = EoutF[keep]
                kmF_phys[keep] = EoutF[keep] * sysP.km_per_kWh
                EinF_alt[sup_flag] = EinF[sup_flag]
                EoutF_alt[sup_flag] = EoutF[sup_flag]
                kmF_alt[sup_flag] = EoutF[sup_flag] * sysP.km_per_kWh
                EinJ2_all = np.zeros(n_exp, dtype=float)
                EoutJ2_all = np.zeros(n_exp, dtype=float)
                kmJ2_all = np.zeros(n_exp, dtype=float)
                retire_year_phys = np.full(n_exp, -1, dtype=np.int32)
                retire_year_phys[keep] = end_keep[keep].astype(np.int32)
                if np.any(sup_flag):
                    sup_idx = np.where(sup_flag)[0]
                    retire_year_phys[sup_idx] = export_year[sup_idx].astype(np.int32)
                total_Ein_service = EinJ + EinJ2_all + EinF_phys + EinF_alt
                total_Eout_service = EoutJ + EoutJ2_all + EoutF_phys + EoutF_alt
                total_km_service = total_Eout_service * sysP.km_per_kWh
                df = pd.DataFrame({'pack_id': pids[idx].astype(int), 'manufacture_year': y, 'route': f"Export_{dest.replace(' ', '_')}", 'jp_years': years_jp[idx].astype(int), 'ov_years': years_ov[idx].astype(int), 'export_year': export_year[idx].astype(int), 'retire_year': retire_year_phys[idx].astype(int), 'captured_network': 0, 'suppressed_export': sup_flag[idx].astype(int), 'Japan_Ein_kWh': EinJ[idx], 'Japan_Eout_kWh': EoutJ[idx], 'Japan_km': EoutJ[idx] * sysP.km_per_kWh, 'Japan_Ein_kWh_additional': EinJ2_all[idx], 'Japan_Eout_kWh_additional': EoutJ2_all[idx], 'Japan_km_additional': kmJ2_all[idx], 'Overseas_Ein_kWh_physical': EinF_phys[idx], 'Overseas_Eout_kWh_physical': EoutF_phys[idx], 'Overseas_km_physical': kmF_phys[idx], 'Overseas_Ein_kWh_altsupply': EinF_alt[idx], 'Overseas_Eout_kWh_altsupply': EoutF_alt[idx], 'Overseas_km_altsupply': kmF_alt[idx], 'Total_Ein_kWh_service': total_Ein_service[idx], 'Total_Eout_kWh_service': total_Eout_service[idx], 'Total_km_service': total_km_service[idx]})
                sample_chunks.append(df)
            pid_base += n_exp
        simulate_export_group(int(alloc_exp['New Zealand']), 'New Zealand')
        simulate_export_group(int(alloc_exp['Mongolia']), 'Mongolia')
        simulate_export_group(int(alloc_exp['Australia']), 'Australia')
        simulate_export_group(int(alloc_exp['Kenya']), 'Kenya')
        simulate_domestic_group(int(alloc['Domestic_BM_JP']), True)
        simulate_domestic_group(int(alloc['Domestic_Disposal_JP']), False)
    assert pid_base == FULL_N, f'pack_id mismatch: {pid_base} != {FULL_N}'
    candidates_by_year_arr = {}
    for yy in range(ip.recovery_start_year, ip.end_year + 1):
        if len(candidates_by_year[yy]) == 0:
            candidates_by_year_arr[yy] = np.zeros((0, 8), dtype=float)
        else:
            candidates_by_year_arr[yy] = np.vstack(candidates_by_year[yy]).astype(float)
    df_bess_annual, hydromet_from_bess_recycled, hydromet_unused_pool, bess_start_actual = simulate_bess_daily_from_candidates(sysP=sysP, mp=mp, ip=ip, candidates_by_year=candidates_by_year_arr, seed=seed)
    active = np.cumsum(diff_active[:, :-1], axis=1)
    deliv = np.cumsum(diff_deliv[:, :-1], axis=1)
    ein = np.cumsum(diff_in[:, :-1], axis=1)
    km = np.cumsum(diff_km[:, :-1], axis=1)
    active_alt = np.cumsum(diff_active_alt[:, :-1], axis=1)
    deliv_alt = np.cumsum(diff_deliv_alt[:, :-1], axis=1)
    ein_alt = np.cumsum(diff_in_alt[:, :-1], axis=1)
    km_alt = np.cumsum(diff_km_alt[:, :-1], axis=1)
    kr_from_jp = bm_jp_outside // 2
    cn_from_jp = bm_jp_outside - kr_from_jp
    hydro_kr = kr_from_jp + bm_au
    hydro_cn = cn_from_jp
    hydromet_japan = np.zeros(nY, dtype=np.int64)
    recovered_to_hydromet_immediate = np.zeros(nY, dtype=np.int64)
    recovered_to_bess = np.zeros(nY, dtype=np.int64)
    bess_retired_packs = np.zeros(nY, dtype=np.int64)
    for yy in range(ip.recovery_start_year, ip.end_year + 1):
        inflow = int(recovered_network_inflow[yy - Y0])
        immediate = int(hydromet_unused_pool.get(yy, 0))
        accepted = max(0, inflow - immediate)
        recovered_to_hydromet_immediate[yy - Y0] = immediate
        recovered_to_bess[yy - Y0] = accepted
    for yy, cnt in hydromet_from_bess_recycled.items():
        yy = int(yy)
        if ip.recovery_start_year <= yy <= ip.end_year:
            bess_retired_packs[yy - Y0] = int(cnt)
    hydromet_japan = recovered_to_hydromet_immediate + bess_retired_packs

    def mats_from_packs(packs):
        packs = packs.astype(float)
        return (packs * ip.REC_NISO4_KG, packs * ip.REC_COSO4_KG, packs * ip.REC_LI2CO3_KG)
    kr_Ni, kr_Co, kr_Li = mats_from_packs(hydro_kr)
    cn_Ni, cn_Co, cn_Li = mats_from_packs(hydro_cn)
    jp_Ni, jp_Co, jp_Li = mats_from_packs(hydromet_japan)
    df_annual = pd.DataFrame({'year': YEARS, 'manufactured_packs': manufactured.astype(int)})
    for c in countries:
        i = c2i[c]
        df_annual[f'active_packs_{c}'] = active[i].astype(int)
        df_annual[f'distance_km_{c}'] = km[i]
        df_annual[f'delivered_kWh_{c}'] = deliv[i]
        df_annual[f'required_input_kWh_{c}'] = ein[i]
        df_annual[f'retired_packs_{c}'] = retired[i].astype(int)
        df_annual[f'active_packs_{c}_altsupply'] = active_alt[i].astype(int)
        df_annual[f'distance_km_{c}_altsupply'] = km_alt[i]
        df_annual[f'delivered_kWh_{c}_altsupply'] = deliv_alt[i]
        df_annual[f'required_input_kWh_{c}_altsupply'] = ein_alt[i]
        df_annual[f'retired_packs_{c}_altsupply'] = retired_alt[i].astype(int)
        df_annual[f'active_packs_{c}_service'] = (active[i] + active_alt[i]).astype(int)
        df_annual[f'distance_km_{c}_service'] = km[i] + km_alt[i]
        df_annual[f'delivered_kWh_{c}_service'] = deliv[i] + deliv_alt[i]
        df_annual[f'required_input_kWh_{c}_service'] = ein[i] + ein_alt[i]
        df_annual[f'retired_packs_{c}_service'] = (retired[i] + retired_alt[i]).astype(int)
    df_annual['bm_processed_packs_Japan_outside'] = bm_jp_outside.astype(int)
    df_annual['disposed_packs_Japan_outside'] = disp_jp_outside.astype(int)
    df_annual['bm_processed_packs_Australia'] = bm_au.astype(int)
    df_annual['bm_packs_Australia_to_Korea'] = bm_au.astype(int)
    df_annual['bm_packs_Japan_to_Korea'] = kr_from_jp.astype(int)
    df_annual['bm_packs_Japan_to_China'] = cn_from_jp.astype(int)
    df_annual['hydromet_packs_Korea'] = hydro_kr.astype(int)
    df_annual['hydromet_packs_China'] = hydro_cn.astype(int)
    df_annual['hydromet_packs_Japan'] = hydromet_japan.astype(int)
    df_annual['NiSO4_kg_Korea'] = kr_Ni
    df_annual['CoSO4_kg_Korea'] = kr_Co
    df_annual['Li2CO3_kg_Korea'] = kr_Li
    df_annual['NiSO4_kg_China'] = cn_Ni
    df_annual['CoSO4_kg_China'] = cn_Co
    df_annual['Li2CO3_kg_China'] = cn_Li
    df_annual['NiSO4_kg_Japan'] = jp_Ni
    df_annual['CoSO4_kg_Japan'] = jp_Co
    df_annual['Li2CO3_kg_Japan'] = jp_Li
    for _c in ('Korea', 'China', 'Japan'):
        df_annual[f'NiSO4_kg_virgin_makeup_{_c}'] = 0.0
        df_annual[f'CoSO4_kg_virgin_makeup_{_c}'] = 0.0
        df_annual[f'Li2CO3_kg_virgin_makeup_{_c}'] = 0.0
    df_annual['recovered_network_inflow_packs_total'] = recovered_network_inflow.astype(int)
    df_annual['recovered_network_inflow_packs_domestic'] = recovered_from_domestic.astype(int)
    df_annual['recovered_network_inflow_packs_suppressed_export'] = recovered_from_suppressed.astype(int)
    df_annual['recovered_trimmed_immediate_hydromet_packs'] = 0
    df_annual['recovered_not_bess_hydromet_packs'] = 0
    df_annual['bess_retired_packs'] = bess_retired_packs.astype(int)
    df_annual['unused_pool_hydromet_packs'] = recovered_to_hydromet_immediate.astype(int)
    df_annual['recovered_to_bess_packs'] = recovered_to_bess.astype(int)
    df_annual['recovered_to_hydromet_immediate_packs'] = recovered_to_hydromet_immediate.astype(int)
    for dest in export_dests:
        df_annual[f"suppressed_exports_{dest.replace(' ', '_')}_packs"] = suppressed_exports_by_dest[dest].astype(int)
        df_annual[f"altsupply_{dest.replace(' ', '_')}_packs"] = alt_supply_by_dest[dest].astype(int)
    df_bess_annual_full = pd.DataFrame({'year': YEARS}).merge(df_bess_annual, on='year', how='left').fillna(0)
    df_sample1000 = pd.concat(sample_chunks, ignore_index=True).sort_values('pack_id') if sample_chunks else pd.DataFrame()
    readme = pd.DataFrame({'note': ['BESS scenario output. (README simplified in split version)']})
    meta = {'FULL_FLEET_total_packs': FULL_N, 'generated_at': jst_now_str(), 'seed': seed, 'ip': ip.__dict__, 'bess_start_actual': str(bess_start_actual) if bess_start_actual is not None else None}
    return (df_annual, df_bess_annual_full, df_sample1000, readme, meta)


# Service-equivalent LFP storage and material make-up

def build_bad_lfp_from_bess_service(df_bess_annual: pd.DataFrame, lp: LFPParams):
    """Size the new LFP comparator for the cumulative service delivered by second-life packs."""
    install_year = 2035
    start_year = 2035
    end_year = 2050
    mask = (df_bess_annual['year'].astype(int) >= start_year) & (df_bess_annual['year'].astype(int) <= end_year)
    if 'LiB_delivered_kWh' in df_bess_annual.columns:
        target_total = float(pd.to_numeric(df_bess_annual.loc[mask, 'LiB_delivered_kWh'], errors='coerce').fillna(0.0).sum())
    else:
        target_total = 0.0
    lfp_add = np.zeros(nY, dtype=np.int64)
    lfp_act = np.zeros(nY, dtype=np.int64)
    lfp_eout = np.zeros(nY, dtype=float)
    lfp_ein = np.zeros(nY, dtype=float)
    lfp_ret = np.zeros(nY, dtype=np.int64)
    lfp_disp = np.zeros(nY, dtype=np.int64)
    if target_total <= 0.0:
        df_lfp = pd.DataFrame({'year': YEARS, 'lfp_additions_modules': lfp_add.astype(int), 'lfp_active_modules': lfp_act.astype(int), 'lfp_delivered_kWh': lfp_eout, 'lfp_required_input_kWh': lfp_ein, 'lfp_retired_modules': lfp_ret.astype(int), 'lfp_disposed_modules': lfp_disp.astype(int), 'lfp_effective_cycles_per_day': lfp_cpd})
        return _append_total_row(df_lfp, 'year', 'TOTAL')
    years_span = end_year - start_year + 1
    days_per_year = 365
    N = years_span * days_per_year
    d = float(lp.dSoH_per_cycle)
    A = float(lp.module_kWh) * float(lp.DoD) * float(lp.eta)
    sum_i = N * (N + 1) / 2.0
    eout_per_module_max = A * (N * 1.0 - d * 1.0 * sum_i)
    modules = int(math.ceil(target_total / max(1e-12, eout_per_module_max)))
    for k in range(years_span):
        yy = start_year + k
        if yy < Y0 or yy > YN:
            continue
        a = k * days_per_year + 1
        b = (k + 1) * days_per_year
        n = days_per_year
        sum_block = (a + b) * n / 2.0
        sumSoH = n - d * sum_block
        Ein_y_1mod = float(lp.module_kWh) * float(lp.DoD) * sumSoH
        Eout_y_1mod = Ein_y_1mod * float(lp.eta)
        lfp_ein[yy - Y0] = float(modules) * Ein_y_1mod
        lfp_eout[yy - Y0] = float(modules) * Eout_y_1mod
    if Y0 <= install_year <= YN:
        lfp_add[install_year - Y0] = int(modules)
    for yy in range(install_year, end_year + 1):
        if Y0 <= yy <= YN:
            lfp_act[yy - Y0] = int(modules)
    if Y0 <= end_year <= YN:
        lfp_ret[end_year - Y0] = int(modules)
        lfp_disp[end_year - Y0] = int(modules)
    df_lfp = pd.DataFrame({'year': YEARS, 'lfp_additions_modules': lfp_add.astype(int), 'lfp_active_modules': lfp_act.astype(int), 'lfp_delivered_kWh': lfp_eout, 'lfp_required_input_kWh': lfp_ein, 'lfp_retired_modules': lfp_ret.astype(int), 'lfp_disposed_modules': lfp_disp.astype(int)})
    return _append_total_row(df_lfp, 'year', 'TOTAL')

def compute_virgin_makeup(df_bad: pd.DataFrame, df_bess: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Calculate primary salt supply needed when overseas recovery is lower than BAU."""

    def _year_index(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if 'year' in d.columns:
            d = d[d['year'].astype(str).str.upper() != 'TOTAL'].copy()
            d['year'] = d['year'].astype(int)
            d = d.set_index('year')
        else:
            d = d[~d.index.astype(str).str.upper().eq('TOTAL')].copy()
            d.index = d.index.astype(int)
        return d.reindex(YEARS).fillna(0.0)
    b = _year_index(df_bad)
    s = _year_index(df_bess)
    out: Dict[str, np.ndarray] = {}
    mats = ['NiSO4', 'CoSO4', 'Li2CO3']
    regions = ['Korea', 'China']
    for reg in regions:
        for mat in mats:
            col = f'{mat}_kg_{reg}'
            if col not in b.columns or col not in s.columns:
                continue
            diff = (b[col] - s[col]).to_numpy(dtype=float)
            out[f'{mat}_kg_virgin_makeup_{reg}'] = np.maximum(diff, 0.0)
    return out
