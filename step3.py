#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: 组装理论预测向量 + χ² 单元测试

依赖: numpy, scipy
运行: python step3.py
确保同目录下有:
  - desi_gaussian_bao_ALL_GCcomb_mean.txt
  - desi_gaussian_bao_ALL_GCcomb_cov.txt
"""

import numpy as np
from scipy.integrate import quad
import os

# =============================================================================
# 常量
# =============================================================================
c_km_s = 299792.458   # 光速 km/s
rd = 147.09           # 声学视界 Mpc

# 文件路径 (同目录)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEAN_FILE = os.path.join(SCRIPT_DIR, 'desi_gaussian_bao_ALL_GCcomb_mean.txt')
COV_FILE  = os.path.join(SCRIPT_DIR, 'desi_gaussian_bao_ALL_GCcomb_cov.txt')

# 类型映射
OBS_MAP  = {'DV_over_rs': 0, 'DM_over_rs': 1, 'DH_over_rs': 2}
OBS_NAME = {0: 'DV_over_rs', 1: 'DM_over_rs', 2: 'DH_over_rs'}

# =============================================================================
# [1] 数据加载
# =============================================================================
def load_data():
    """加载 mean 和 cov 文件, 返回 z_arr, value_arr, type_arr, C"""
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
    value_arr = np.array(val_list)
    type_arr = np.array(typ_list)
    n = len(z_arr)

    with open(COV_FILE) as f:
        tokens = f.read().split()
    C = np.array([float(t) for t in tokens]).reshape(n, n)

    return z_arr, value_arr, type_arr, C

# =============================================================================
# [2] H(z) 双幂律 + 距离函数
# =============================================================================
def H_z(z, H0, alpha, beta, gamma, delta):
    """
    H(z) = H0 * [ ((1+z)^beta + alpha*(1+z)^gamma) / (1+alpha) ]^delta
    """
    zp1 = 1.0 + np.asarray(z, dtype=float)
    bracket = (zp1**beta + alpha * zp1**gamma) / (1.0 + alpha)
    return H0 * np.power(bracket, delta)


def DH_over_rd(z, H0, alpha, beta, gamma, delta):
    """DH/rd = c / (H(z) * rd)"""
    return c_km_s / (H_z(z, H0, alpha, beta, gamma, delta) * rd)


def DM_over_rd(z, H0, alpha, beta, gamma, delta):
    """DM/rd = (c/rd) * ∫_0^z dz'/H(z')"""
    integral, _ = quad(
        lambda zp: 1.0 / H_z(zp, H0, alpha, beta, gamma, delta),
        0, float(z), epsabs=1e-10, epsrel=1e-10, limit=200
    )
    return (c_km_s / rd) * integral


def DV_over_rd(z, H0, alpha, beta, gamma, delta):
    """DV/rd = [z * (DM)^2 * (DH)]^{1/3} / rd"""
    dm = DM_over_rd(z, H0, alpha, beta, gamma, delta)
    dh = DH_over_rd(z, H0, alpha, beta, gamma, delta)
    return (z * (dm * rd)**2 * (dh * rd)) ** (1.0 / 3.0) / rd

# =============================================================================
# [3] 组装理论预测向量
# =============================================================================
def compute_theory(params, z_arr, type_arr):
    """
    按 type_arr 顺序组装 13 维理论预测向量。
    type: 0=DV/rd, 1=DM/rd, 2=DH/rd
    """
    H0, alpha, beta, gamma, delta = params
    theory = np.zeros(len(z_arr))
    for i in range(len(z_arr)):
        z = z_arr[i]
        if type_arr[i] == 0:
            theory[i] = DV_over_rd(z, H0, alpha, beta, gamma, delta)
        elif type_arr[i] == 1:
            theory[i] = DM_over_rd(z, H0, alpha, beta, gamma, delta)
        elif type_arr[i] == 2:
            theory[i] = DH_over_rd(z, H0, alpha, beta, gamma, delta)
    return theory

# =============================================================================
# [4] χ² 计算函数
# =============================================================================
def chi2_func(theory, obs, inv_cov):
    """χ² = (obs - theory)^T · C^{-1} · (obs - theory)"""
    diff = obs - theory
    return float(diff @ inv_cov @ diff)


def chi2_from_params(params, z_arr, type_arr, obs, inv_cov):
    """从参数直接算 χ² (便捷接口)"""
    theory = compute_theory(params, z_arr, type_arr)
    return chi2_func(theory, obs, inv_cov)

# =============================================================================
# 主程序
# =============================================================================
if __name__ == '__main__':

    # ------------------------------------------------------------------
    # 加载数据
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  [1] 数据加载")
    print("=" * 70)
    z_arr, value_arr, type_arr, C = load_data()
    n = len(z_arr)
    inv_C = np.linalg.inv(C)
    sigma_arr = np.sqrt(np.diag(C))
    print(f"  >> {n} 个数据点, 协方差 {C.shape[0]}x{C.shape[1]}")

    # ------------------------------------------------------------------
    # [A] 理论预测向量
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  [A] 理论预测向量  params=(67.4, 1, 1.5, 0, 1)")
    print("=" * 70)

    params_test = [67.4, 1.0, 1.5, 0.0, 1.0]
    theory = compute_theory(params_test, z_arr, type_arr)

    print(f"\n  {'#':>3}  {'z':>6}  {'类型':<12}  {'理论值':>14}")
    print("  " + "-" * 45)
    for i in range(n):
        print(f"  {i+1:>3}  {z_arr[i]:>6.3f}  {OBS_NAME[type_arr[i]]:<12}  {theory[i]:>14.6f}")

    ok12 = (type_arr[11] == 2)
    ok13 = (type_arr[12] == 1)
    range_ok = all(3 < v < 50 for v in theory)
    print(f"\n  第12点: {OBS_NAME[type_arr[11]]} {'✓ DH' if ok12 else '✗'}")
    print(f"  第13点: {OBS_NAME[type_arr[12]]} {'✓ DM' if ok13 else '✗'}")
    print(f"  数值范围合理 (DH~10-30, DM~10-40, DV~10-30): {'✓' if range_ok else '✗'}")

    # ------------------------------------------------------------------
    # [B] χ² 单元测试
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  [B] χ² 单元测试")
    print("=" * 70)

    # 动态计算期望值
    # 注: 协方差矩阵为块对角结构 (同红移 DM-DH 有反相关),
    #     因此 σ^T C^{-1} σ ≠ 13 (对角时才是13)
    #     这是数学上严谨的结果, 用于验证 χ² 函数实现正确
    chi2_sigma_expected = float(sigma_arr @ inv_C @ sigma_arr)
    chi2_01sigma_expected = 0.01 * chi2_sigma_expected

    print(f"\n  期望值 (从实际协方差矩阵动态计算):")
    print(f"    χ²(obs+σ)      = {chi2_sigma_expected:.6f}")
    print(f"      注: 若C为对角矩阵则=13.0, 但实际DM-DH有反相关(ρ≈-0.3~-0.5)")
    print(f"      同红移2×2块: χ² = 2/(1+ρ), 反相关使 ρ<0 → χ² > 2")
    print(f"    χ²(obs+0.1σ)   = {chi2_01sigma_expected:.6f}")
    print(f"      注: 若C为对角矩阵则=0.13")

    # 测试 1: 完美拟合 → χ² = 0
    t1 = chi2_func(value_arr, value_arr, inv_C)
    t1_pass = abs(t1) < 1e-6
    print(f"\n  测试1: theo = obs (完美拟合)")
    print(f"    χ² = {t1:.10f}, 期望 = 0  → {'PASS' if t1_pass else 'FAIL'}")

    # 测试 2: 偏移 1σ
    t2 = chi2_func(value_arr + sigma_arr, value_arr, inv_C)
    t2_pass = abs(t2 - chi2_sigma_expected) < 1e-4
    print(f"\n  测试2: theo = obs + σ")
    print(f"    χ² = {t2:.6f}, 期望 = {chi2_sigma_expected:.6f}  → {'PASS' if t2_pass else 'FAIL'}")

    # 测试 3: 偏移 0.1σ
    t3 = chi2_func(value_arr + 0.1 * sigma_arr, value_arr, inv_C)
    t3_pass = abs(t3 - chi2_01sigma_expected) < 1e-5
    print(f"\n  测试3: theo = obs + 0.1σ")
    print(f"    χ² = {t3:.6f}, 期望 = {chi2_01sigma_expected:.6f}  → {'PASS' if t3_pass else 'FAIL'}")

    # 各块贡献明细
    print(f"\n  (附) 各 2×2 块的 χ² 贡献明细 (测试2):")
    print(f"  {'块':>10}  {'类型':<25}  {'ρ':>8}  {'χ²块':>10}  {'解析 2/(1+ρ)':>14}")
    print("  " + "-" * 75)
    blocks = [(0,), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
    for idxs in blocks:
        d = sigma_arr[list(idxs)]
        sub_inv = np.linalg.inv(C[np.ix_(idxs, idxs)])
        block_chi2 = float(d @ sub_inv @ d)
        if len(idxs) == 2:
            i, j = idxs
            rho = C[i, j] / (sigma_arr[i] * sigma_arr[j])
            analytic = 2.0 / (1.0 + rho)
            names = f"{OBS_NAME[type_arr[i]]}, {OBS_NAME[type_arr[j]]}"
            print(f"  {str(idxs):>10}  {names:<25}  {rho:>8.4f}  {block_chi2:>10.4f}  {analytic:>14.4f}")
        else:
            i = idxs[0]
            print(f"  {str(idxs):>10}  {OBS_NAME[type_arr[i]]:<25}  {'—':>8}  {block_chi2:>10.4f}  {'—':>14}")

    # ------------------------------------------------------------------
    # [C] 总结
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  [C] 验证总结")
    print("=" * 70)

    checks = [
        ("第12点 = DH_over_rs",      ok12),
        ("第13点 = DM_over_rs",      ok13),
        ("理论值范围合理",            range_ok),
        ("χ²测试1: 完美拟合 = 0",    t1_pass),
        ("χ²测试2: obs+σ",           t2_pass),
        ("χ²测试3: obs+0.1σ",        t3_pass),
    ]
    for desc, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {desc}")

    print(f"\n  常数: c = {c_km_s} km/s, rd = {rd} Mpc")
    if all(r for _, r in checks):
        print("\n  ==============================")
        print("   ALL CHECKS PASSED!")
        print("  ==============================")
    else:
        print("\n  ⚠️  部分验证未通过！")
    print("=" * 70)
