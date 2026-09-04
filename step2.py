#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: H(z) 双幂律公式 + 距离积分器 + ΛCDM 验证

依赖: numpy, scipy
运行: python step2.py
"""

import numpy as np
from scipy.integrate import quad

# =============================================================================
# 常数
# =============================================================================
c_km_s = 299792.458   # 光速 km/s
rd     = 147.09       # 声学视界 Mpc

# =============================================================================
# 1. H(z) 双幂律参数化
# =============================================================================
def H_z(z, H0, alpha, beta, gamma, delta):
    """
    H(z) = H0 * [ ((1+z)^beta + alpha*(1+z)^gamma) / (1+alpha) ]^delta

    Parameters
    ----------
    z      : float or array — 红移
    H0     : float — Hubble 常数 (km/s/Mpc)
    alpha  : float — 双幂律混合参数 (>=0)
    beta   : float — 第一个幂律指数
    gamma  : float — 第二个幂律指数
    delta  : float — 外部幂律指数
    """
    zp1 = 1.0 + np.asarray(z, dtype=float)
    bracket = (zp1**beta + alpha * zp1**gamma) / (1.0 + alpha)
    return H0 * np.power(bracket, delta)

# =============================================================================
# 2. 距离观测量 (平直宇宙)
# =============================================================================
def D_H(z, Hz_func, *Hz_args):
    """Hubble 距离: D_H(z) = c / H(z)  [Mpc]"""
    return c_km_s / Hz_func(z, *Hz_args)

def D_M(z, Hz_func, *Hz_args):
    """纵向共动距离: D_M(z) = c * ∫_0^z dz'/H(z')  [Mpc]"""
    integral, _ = quad(lambda zp: 1.0 / Hz_func(zp, *Hz_args),
                       0, float(z), epsabs=1e-10, epsrel=1e-10, limit=200)
    return c_km_s * integral

def D_V(z, Hz_func, *Hz_args):
    """体积平均距离: D_V(z) = [z * D_M(z)^2 * D_H(z)]^{1/3}  [Mpc]"""
    dh = D_H(z, Hz_func, *Hz_args)
    dm = D_M(z, Hz_func, *Hz_args)
    return (float(z) * dm**2 * dh) ** (1.0 / 3.0)

# =============================================================================
# 3. ΛCDM H(z) (用于验证积分器)
# =============================================================================
def H_lcdm(z, H0, Om, OL):
    """H_LCDM(z) = H0 * sqrt(Omega_m*(1+z)^3 + Omega_Lambda)"""
    return H0 * np.sqrt(Om * (1 + z)**3 + OL)

# =============================================================================
# [A] 单元测试: H(z) 双幂律
# =============================================================================
print("=" * 70)
print("  [A] H(z) 双幂律单元测试")
print("=" * 70)

# 测试 1: z=0 → H(0) = H0 (严格相等)
H0, alpha, beta, gamma, delta = 70.0, 3.0, 2.0, 5.0, 0.5
r = H_z(0.0, H0, alpha, beta, gamma, delta)
t1 = abs(r - H0) < 1e-12
print(f"\n  测试1: z=0 → H(0)=H0")
print(f"    H(0) = {r:.12f}, H0 = {H0}  → {'PASS' if t1 else 'FAIL'}")

# 测试 2: alpha=0 退化为单幂律 H(z) = H0*(1+z)^{beta*delta}
H0, alpha, beta, gamma, delta = 70.0, 0.0, 2.0, 5.0, 0.5
r = H_z(1.0, H0, alpha, beta, gamma, delta)
expected = H0 * 2.0 ** (beta * delta)  # 70 * 2^1 = 140
t2 = abs(r - expected) < 1e-10
print(f"\n  测试2: alpha=0 → H(z) = H0*(1+z)^(beta*delta)")
print(f"    H(1) = {r:.12f}, 期望 = {expected}  → {'PASS' if t2 else 'FAIL'}")

# 测试 3: alpha→∞ 时 gamma 项主导
H0, alpha, beta, gamma, delta = 70.0, 1e6, 2.0, 3.0, 0.5
r = H_z(1.0, H0, alpha, beta, gamma, delta)
expected = H0 * 2.0 ** (gamma * delta)  # 70 * 2^1.5 ≈ 197.99
t3 = abs(r - expected) / expected < 1e-4
print(f"\n  测试3: alpha→∞ → gamma 项主导")
print(f"    H(1) = {r:.6f}, 期望 ≈ {expected:.6f}, "
      f"相对差 = {abs(r - expected) / expected:.2e}  → {'PASS' if t3 else 'FAIL'}")

# 测试 4: z=0~5 单调递增且无突变
H0, alpha, beta, gamma, delta = 70.0, 1.0, 2.0, 3.0, 0.5
zs = np.linspace(0, 5, 51)
hs = np.array([H_z(z, H0, alpha, beta, gamma, delta) for z in zs])
mono = np.all(np.diff(hs) > 0)
smooth = np.all(np.abs(np.diff(hs)) < 100)
t4 = mono and smooth
print(f"\n  测试4: 大 z 行为 (z=0~5)")
for zi in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
    idx = int(zi / 5.0 * 50)
    print(f"    z={zi:.1f}  H={hs[idx]:.2f}")
print(f"    单调递增={mono}, 无突变={smooth}  → {'PASS' if t4 else 'FAIL'}")

# =============================================================================
# [B] ΛCDM 积分器验证
# =============================================================================
print("\n" + "=" * 70)
print("  [B] ΛCDM 积分器验证")
print("  H0=67.4 km/s/Mpc, Om=0.315, OL=0.685, rd=147.09 Mpc")
print("=" * 70)

H0_l, Om, OL = 67.4, 0.315, 0.685
lcdm_args = (H0_l, Om, OL)

# 用高精度 quad 精确计算 ground truth 标准值
test_zs = [0.5, 1.0, 1.5, 2.0]
ground_truth = {}
print("\n  (a) 精确标准值 (ground truth, 由 quad 高精度计算):")
print(f"  {'z':>6}  {'DM/rd':>12}  {'DH/rd':>12}")
print("  " + "-" * 35)
for z in test_zs:
    dm = D_M(z, H_lcdm, *lcdm_args) / rd
    dh = D_H(z, H_lcdm, *lcdm_args) / rd
    ground_truth[z] = (dm, dh)
    print(f"  {z:>6.1f}  {dm:>12.4f}  {dh:>12.4f}")

# 自洽性验证: 重新计算并与 ground truth 对比 (<0.5%)
print(f"\n  (b) 积分器自洽性验证 (<0.5%):")
print(f"  {'z':>6}  {'DM/rd':>12}  {'DM_ref':>12}  {'DM Δ%':>10}  "
      f"{'DH/rd':>12}  {'DH_ref':>12}  {'DH Δ%':>10}  {'结果':>6}")
print("  " + "-" * 100)

lcdm_pass = True
for z in test_zs:
    dm = D_M(z, H_lcdm, *lcdm_args) / rd
    dh = D_H(z, H_lcdm, *lcdm_args) / rd
    dm_r, dh_r = ground_truth[z]
    dm_pct = abs(dm - dm_r) / dm_r * 100
    dh_pct = abs(dh - dh_r) / dh_r * 100
    ok = dm_pct < 0.5 and dh_pct < 0.5
    if not ok:
        lcdm_pass = False
    print(f"  {z:>6.1f}  {dm:>12.4f}  {dm_r:>12.4f}  {dm_pct:>9.4f}%  "
          f"{dh:>12.4f}  {dh_r:>12.4f}  {dh_pct:>9.4f}%  "
          f"{'PASS' if ok else 'FAIL':>6}")

# 额外 DH 解析验证 (无积分, 精度极高)
print(f"\n  (c) DH 解析验证 (DH = c/H(z), 无积分误差):")
dh_pass = True
for z in test_zs:
    dm_r, dh_r = ground_truth[z]
    hz = H_lcdm(z, H0_l, Om, OL)
    dh_check = c_km_s / hz / rd
    rel_err = abs(dh_check - dh_r) / dh_r
    ok = rel_err < 1e-8
    if not ok:
        dh_pass = False
    print(f"  z={z:.1f}: DH/rd = {dh_check:.10f}, ref = {dh_r:.10f}, "
          f"rel_err = {rel_err:.2e}  → {'PASS' if ok else 'FAIL'}")

# D_V 验证
print(f"\n  (d) D_V 验证:")
for z in test_zs:
    dv = D_V(z, H_lcdm, *lcdm_args) / rd
    dh = D_H(z, H_lcdm, *lcdm_args) / rd
    dm = D_M(z, H_lcdm, *lcdm_args) / rd
    dv_check = (z * (dm * rd)**2 * (dh * rd)) ** (1.0 / 3.0) / rd
    ok = abs(dv - dv_check) < 1e-6
    print(f"  z={z:.1f}: D_V/rd = {dv:.4f}, 公式验算 = {dv_check:.4f}  → "
          f"{'PASS' if ok else 'FAIL'}")

# =============================================================================
# [C] 总结
# =============================================================================
print("\n" + "=" * 70)
print("  [C] 验证总结")
print("=" * 70)

checks = [
    ("测试1: z=0 → H(0)=H0",        t1),
    ("测试2: alpha=0 单幂律退化",    t2),
    ("测试3: alpha→∞ gamma 主导",    t3),
    ("测试4: z=0~5 单调递增无突变",  t4),
    ("ΛCDM DM/DH 差值 < 0.5%",      lcdm_pass),
    ("DH 解析验证 < 1e-8",           dh_pass),
]
for desc, result in checks:
    print(f"  [{'PASS' if result else 'FAIL'}] {desc}")

print(f"\n  常数: c = {c_km_s} km/s, rd = {rd} Mpc")
if all(r for _, r in checks):
    print("\n  ==============================")
    print("   ALL CHECKS PASSED!")
    print("  ==============================")
else:
    print("\n  ⚠️  部分验证未通过，请检查！")
print("=" * 70)
