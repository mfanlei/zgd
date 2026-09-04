#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: 多起始点 L-BFGS-B 拟合双幂律参数
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import os
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 常量
# =============================================================================
c_km_s = 299792.458
rd = 147.09

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEAN_FILE = os.path.join(SCRIPT_DIR, 'desi_gaussian_bao_ALL_GCcomb_mean.txt')
COV_FILE  = os.path.join(SCRIPT_DIR, 'desi_gaussian_bao_ALL_GCcomb_cov.txt')

OBS_MAP  = {'DV_over_rs': 0, 'DM_over_rs': 1, 'DH_over_rs': 2}
OBS_NAME = {0: 'DV_over_rs', 1: 'DM_over_rs', 2: 'DH_over_rs'}

# =============================================================================
# [1] 数据加载
# =============================================================================
def load_data():
    z_list, val_list, typ_list = [], [], []
    with open(MEAN_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split()
            z_list.append(float(p[0]))
            val_list.append(float(p[1]))
            typ_list.append(OBS_MAP[p[2]])
    z_arr = np.array(z_list)
    val_arr = np.array(val_list)
    typ_arr = np.array(typ_list)
    n = len(z_arr)
    with open(COV_FILE) as f:
        tokens = f.read().split()
    C = np.array([float(t) for t in tokens]).reshape(n, n)
    return z_arr, val_arr, typ_arr, C

# =============================================================================
# [2] H(z) + 距离函数
# =============================================================================
def H_z(z, H0, alpha, beta, gamma, delta):
    zp1 = 1.0 + np.asarray(z, dtype=float)
    bracket = (zp1**beta + alpha * zp1**gamma) / (1.0 + alpha)
    return H0 * np.power(bracket, delta)

def DH_over_rd(z, params):
    return c_km_s / (H_z(z, *params) * rd)

def DM_over_rd(z, params):
    integral, _ = quad(
        lambda zp: 1.0 / H_z(zp, *params),
        0, float(z), epsabs=1e-10, epsrel=1e-10, limit=200
    )
    return (c_km_s / rd) * integral

def DV_over_rd(z, params):
    dm = DM_over_rd(z, params)
    dh = DH_over_rd(z, params)
    return (z * (dm * rd)**2 * (dh * rd)) ** (1.0 / 3.0) / rd

# =============================================================================
# [3] 理论向量 + chi2
# =============================================================================
def compute_theory(params, z_arr, typ_arr, unique_z):
    n = len(z_arr)
    dh_cache = {}
    dm_cache = {}
    for z in unique_z:
        dh_cache[z] = float(DH_over_rd(z, params))
        dm_cache[z] = float(DM_over_rd(z, params))
    theory = np.zeros(n)
    for i in range(n):
        z = z_arr[i]
        if typ_arr[i] == 0:
            dm = dm_cache[z]
            dh = dh_cache[z]
            theory[i] = (z * (dm * rd)**2 * (dh * rd)) ** (1.0/3.0) / rd
        elif typ_arr[i] == 1:
            theory[i] = dm_cache[z]
        elif typ_arr[i] == 2:
            theory[i] = dh_cache[z]
    return theory

def chi2_func(params, z_arr, typ_arr, val_arr, inv_C, unique_z):
    try:
        theory = compute_theory(params, z_arr, typ_arr, unique_z)
        diff = val_arr - theory
        result = float(diff @ inv_C @ diff)
        return result if np.isfinite(result) else 1e10
    except Exception:
        return 1e10

# =============================================================================
# [4] 物理合理性检查
# =============================================================================
def is_physical(p, verbose=False):
    H0, alpha, beta, gamma, delta = p
    if H0 < 55 or H0 > 85:
        if verbose: print(f"    !! H0={H0:.2f} out of [55,85]")
        return False
    if alpha < 0.005 or alpha > 5000:
        if verbose: print(f"    !! alpha={alpha:.4e} near degenerate limit")
        return False
    if abs(beta) > 80 or abs(gamma) > 80:
        if verbose: print(f"    !! beta={beta:.2f} or gamma={gamma:.2f} too large")
        return False
    if delta < 0.02:
        if verbose: print(f"    !! delta={delta:.6f} too small")
        return False
    eff = delta * max(beta, gamma)
    if eff < 0.3 or eff > 5.0:
        if verbose: print(f"    !! delta*max(beta,gamma)={eff:.4f} out of [0.3,5]")
        return False
    return True

# =============================================================================
# [5] 多起始点拟合
# =============================================================================
def multi_start_fit(z_arr, val_arr, typ_arr, C):
    n = len(z_arr)
    inv_C = np.linalg.inv(C)
    unique_z = sorted(set(z_arr))
    sigma_arr = np.sqrt(np.diag(C))

    bounds = [(50, 90), (0.01, 1000), (1.01, 50), (-5, 50), (0.01, 5)]

    def objective(params):
        return chi2_func(params, z_arr, typ_arr, val_arr, inv_C, unique_z)

    starts = [
        [67.4, 1,     1.5,  0,    1   ],
        [65,   5,     10,   30,   0.17],
        [70,   0.5,   2,   -1,    0.8 ],
        [68,   500,   20,   10,   0.08],
        [72,   10,    3,    35,   0.15],
        [66,   100,   5,    20,   0.1 ],
        [70,   1,     15,   40,   0.05],
        [65,   0.1,   2,    5,    0.5 ],
        [68,   3,     8,    25,   0.2 ],
        [71,   50,    12,   15,   0.12],
        [64,   200,   25,   8,    0.07],
        [73,   2,     3,   -2,    1.5 ],
        [69,   800,   30,   5,    0.04],
        [67,   0.05,  1.5,  50,   0.3 ],
        [70,   20,    7,    20,   0.1 ],
    ]

    # ----- [A] L-BFGS-B -----
    print("=" * 70)
    print("  [A] L-BFGS-B (15 starts)")
    print("=" * 70)
    results = []
    t0 = time.time()
    for idx, x0 in enumerate(starts):
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            results.append((res.fun, np.array(res.x), res.success, idx))
            tag = 'OK' if res.success else '!!'
            phys = is_physical(res.x)
            pstr = ', '.join(f'{v:.4f}' for v in res.x)
            print(f"  #{idx+1:>2d}  chi2={res.fun:>10.4f}  {tag:>4}  {'Y' if phys else 'N':>2}  [{pstr}]")
        except Exception as e:
            print(f"  #{idx+1:>2d}  FAIL {e}")
            results.append((1e10, np.array(x0), False, idx))
    elapsed = time.time() - t0
    n_ok = sum(1 for _, _, c, _ in results if c)
    print(f"\n  time: {elapsed:.1f}s  success: {n_ok}/{len(starts)}")

    # 从物理合理的解中选最佳
    valid_phys = sorted(
        [(c2, x, i) for c2, x, c, i in results
         if np.isfinite(c2) and c2 < 1e8 and is_physical(x)],
        key=lambda t: t[0]
    )
    valid_all = sorted(
        [(c2, x, i) for c2, x, c, i in results
         if np.isfinite(c2) and c2 < 1e8],
        key=lambda t: t[0]
    )

    if valid_phys:
        best_c2, best_x, best_idx = valid_phys[0]
        print(f"\n  Best (physical): start #{best_idx+1}, chi2 = {best_c2:.6f}")
    else:
        best_c2, best_x, best_idx = valid_all[0]
        print(f"\n  Best (no filter): start #{best_idx+1}, chi2 = {best_c2:.6f}")
    pstr = ', '.join(f'{v:.6f}' for v in best_x)
    print(f"    params: [{pstr}]")

    # ----- [B] Powell (带物理检查) -----
    print(f"\n{'='*70}")
    print("  [B] Powell refinement (with physical check)")
    print("=" * 70)
    try:
        res_p = minimize(objective, best_x, method='Powell',
                         options={'maxiter': 5000, 'ftol': 1e-14})
        p_converged = res_p.success
        p_chi2 = res_p.fun
        p_phys = is_physical(res_p.x, verbose=True)
        pstr_p = ', '.join(f'{v:.6f}' for v in res_p.x)
        print(f"\n  Powell chi2 = {p_chi2:.6f}  {'OK' if p_converged else '!!'}")
        print(f"  params: [{pstr_p}]")
        print(f"  physical: {'YES' if p_phys else 'NO'}")

        if p_converged and p_phys and p_chi2 < best_c2:
            final_params = np.array(res_p.x)
            final_chi2 = p_chi2
            print("  -> Powell is better and physical, use Powell")
        else:
            final_params = best_x.copy()
            final_chi2 = best_c2
            if not p_phys:
                print("  -> Powell params NOT physical (alpha/delta degenerate), keep L-BFGS-B")
            else:
                print(f"  -> L-BFGS-B is better ({best_c2:.4f} vs {p_chi2:.4f}), keep L-BFGS-B")
    except Exception as e:
        final_params = best_x.copy()
        final_chi2 = best_c2
        print(f"  Powell failed: {e}, keep L-BFGS-B")

    # ----- [C] 诊断 -----
    print(f"\n{'='*70}")
    print("  [C] Diagnostics")
    print("=" * 70)
    ndof = n - 5
    H0, alpha, beta, gamma, delta = final_params

    print(f"\n  chi2 = {final_chi2:.4f},  ndof = {ndof},  chi2/ndof = {final_chi2/ndof:.4f}")
    print(f"\n  Best-fit params:")
    print(f"    H0    = {H0:.6f} km/s/Mpc")
    print(f"    alpha = {alpha:.6f}")
    print(f"    beta  = {beta:.6f}")
    print(f"    gamma = {gamma:.6f}")
    print(f"    delta = {delta:.6f}")
    print(f"    beta*delta  = {beta*delta:.6f}")
    print(f"    gamma*delta = {gamma*delta:.6f}")

    phys = is_physical(final_params, verbose=True)

    z_check = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    h_check = [float(H_z(z, *final_params)) for z in z_check]
    monotonic = all(h_check[i] < h_check[i+1] for i in range(len(h_check)-1))
    print(f"\n  H(z) monotonic:")
    for z, h in zip(z_check, h_check):
        print(f"    H({z:.1f}) = {h:.2f} km/s/Mpc")
    print(f"    monotonic: {'YES' if monotonic else 'NO'}")

    print(f"\n  Degenerate candidates (dchi2 < 2, physical only):")
    deg_count = 0
    for c2, x, i in valid_phys:
        if 0 < (c2 - final_chi2) < 2:
            pstr = ', '.join(f'{v:.4f}' for v in x)
            print(f"    #{i+1}  chi2={c2:.4f}  [{pstr}]")
            deg_count += 1
    if deg_count == 0:
        print(f"    None")

    # ----- [D] 残差表 -----
    print(f"\n{'='*70}")
    print("  [D] Residuals")
    print("=" * 70)
    theory_final = compute_theory(final_params, z_arr, typ_arr, unique_z)
    print(f"\n  {'#':>3}  {'z':>6}  {'type':<12}  {'obs':>10}  {'theo':>10}  {'res':>10}  {'r/s':>8}")
    print("  " + "-" * 70)
    all_in_3sigma = True
    for i in range(n):
        resid = val_arr[i] - theory_final[i]
        resid_sigma = resid / sigma_arr[i]
        ok = abs(resid_sigma) < 3
        if not ok:
            all_in_3sigma = False
        print(f"  {i+1:>3}  {z_arr[i]:>6.3f}  {OBS_NAME[typ_arr[i]]:<12}  "
              f"{val_arr[i]:>10.4f}  {theory_final[i]:>10.4f}  "
              f"{resid:>+10.4f}  {resid_sigma:>+8.3f}")
    print(f"\n  All residuals in +/-3 sigma: {'YES' if all_in_3sigma else 'NO'}")

    # ----- [E] 总结 -----
    print(f"\n{'='*70}")
    print("  [E] Summary")
    print("=" * 70)
    checks = [
        ("chi2/ndof < 1.5",  final_chi2 / ndof < 1.5),
        ("params physical",   phys),
        ("residuals +/-3sigma", all_in_3sigma),
        ("H(z) monotonic",    monotonic),
    ]
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
    print(f"\n  c = {c_km_s} km/s, rd = {rd} Mpc")
    if all(ok for _, ok in checks):
        print("\n  ==============================")
        print("   ALL CHECKS PASSED!")
        print("  ==============================")
    else:
        print("\n  WARNING: some checks failed")
    print("=" * 70)
    return final_params, final_chi2

# =============================================================================
# main
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("  [0] Load data")
    print("=" * 70)
    z_arr, val_arr, typ_arr, C = load_data()
    n = len(z_arr)
    print(f"  >> {n} data points, cov {C.shape[0]}x{C.shape[1]}")
    ok12 = (typ_arr[11] == 2)
    ok13 = (typ_arr[12] == 1)
    print(f"  >> point 12: {OBS_NAME[typ_arr[11]]} {'OK' if ok12 else 'FAIL'}")
    print(f"  >> point 13: {OBS_NAME[typ_arr[12]]} {'OK' if ok13 else 'FAIL'}")
    unique_z = sorted(set(z_arr))
    print(f"  >> unique z: {unique_z} ({len(unique_z)} points)")
    final_params, final_chi2 = multi_start_fit(z_arr, val_arr, typ_arr, C)
