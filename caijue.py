#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# verify_bare_lambda.py —— 论文一附录F/G 数值复现（对应修订定稿）
# 用法: 与 desi_gaussian_bao_ALL_GCcomb_mean.txt / _cov.txt 放同一目录运行
#
# 验证目标:
#   附录F  : 裸几何 χ²=1790.0 (/13≈137.7); λ*=1.1248, χ²=187.1 (/12≈15.6);
#            残差结构: 11/13点 |r/σ|≤1.5; z=0.934 D_H ≈ +4.5σ;
#            z=2.33 D_H ≈ −10.7σ (与裸几何预言一致 0.06σ);
#            "λ≈1.14" 为剔除 z=2.33 D_H 点后的12点拟合
#   附录G表8: 三模型 χ² 对比 1790.0 / 187.1 / 5.50
#   附录G表9: D_H/r_d 预言四列 (z=1.5, 2.0, 2.5, 3.0)
import numpy as np

c_km_s, rd, H0_BARE = 299792.458, 147.09, 70.85
OBS_MAP = {'DV_over_rs': 0, 'DM_over_rs': 1, 'DH_over_rs': 2}
OBS_NAME = {0: 'DV/rd', 1: 'DM/rd', 2: 'DH/rd'}

z_l, v_l, t_l = [], [], []
for line in open('desi_gaussian_bao_ALL_GCcomb_mean.txt'):
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    p = line.split()
    z_l.append(float(p[0])); v_l.append(float(p[1])); t_l.append(OBS_MAP[p[2]])
z = np.array(z_l); obs = np.array(v_l); typ = np.array(t_l); n = len(z)
C = np.array([float(t) for t in open('desi_gaussian_bao_ALL_GCcomb_cov.txt').read().split()]).reshape(n, n)
Cinv = np.linalg.inv(C)
print(f"数据点 {n}，协方差 {C.shape}，对称={np.allclose(C, C.T)}")

def bare_theory(zv, H0):
    """coasting 解析理论: H=H0(1+z), 无积分"""
    K = c_km_s / (H0 * rd)
    DH = K / (1.0 + zv); DM = K * np.log(1.0 + zv)
    DV = (zv * DM**2 * DH) ** (1.0/3.0)
    return np.where(typ == 0, DV, np.where(typ == 1, DM, DH))

chi2 = lambda t: float((obs - t) @ Cinv @ (obs - t))

# ---------------------------------------------------------------
# (a)(b)(c) 裸几何与 λ 模型 → 附录F / 附录G表8
# ---------------------------------------------------------------
t_bare = bare_theory(z, H0_BARE)
print(f"\n(a) 裸几何 H0=70.85, λ=1 :  χ²={chi2(t_bare):9.2f}  /13={chi2(t_bare)/13:6.2f}"
      f"   [附录F/表8: 1790.0, 137.7]")

a_q = t_bare @ Cinv @ t_bare; b_q = t_bare @ Cinv @ obs; d_q = obs @ Cinv @ obs
lam_star = b_q / a_q; c2min = d_q - b_q**2 / a_q
print(f"(b) λ=1.1248             :  χ²={chi2(t_bare*1.1248):9.2f}  /12={chi2(t_bare*1.1248)/12:6.2f}"
      f"   [附录F/表8: 187.1, 15.6]")
print(f"(c) λ最优 = {lam_star:.4f}      :  χ²min={c2min:9.2f}  /12={c2min/12:6.2f}"
      f"   [附录F: λ*=1.1248]")

# (c2) 剔除 z=2.33 的 D_H 点(第12点)后的12点拟合 → 附录F括注 "λ≈1.14"
mask = np.ones(n, dtype=bool); mask[11] = False
t12, o12 = t_bare[mask], obs[mask]
C12inv = np.linalg.inv(C[np.ix_(mask, mask)])
lam12 = (t12 @ C12inv @ o12) / (t12 @ C12inv @ t12)
print(f"(c2) 剔除 z=2.33 D_H 后12点拟合: λ = {lam12:.4f}   [附录F括注: λ≈1.14]")

# (d) 恒等性检验: coasting 下所有距离 ∝ 1/H0, 故 λ 模型 ≡ 释放 H0 模型
H0_free = H0_BARE / lam_star
c2_h0 = chi2(bare_theory(z, H0_free))
print(f"(d) 释放H0路径: H0*={H0_free:.2f}, χ²={c2_h0:.2f}  "
      f"与(c)之差={abs(c2_h0-c2min):.1e} (应≈0: λ模型≡释放H0, 同一模型族)")

# ---------------------------------------------------------------
# (e)(g) 双幂律与表9预言 → 附录G表8/表9 (需要 scipy)
# ---------------------------------------------------------------
try:
    from scipy.integrate import quad
    def H_z(zp, H0, al, be, ga, de):
        z1 = 1.0 + zp
        return H0 * ((z1**be + al*z1**ga) / (1.0 + al))**de
    # 表4为四位舍入值; 此处用 step4 全精度最优拟合以精确复现 5.50
    DPL = (64.841464, 1000.0, 22.257804, 13.225237, 0.064965)
    def theory_dpl(pr):
        out = np.zeros(n)
        for i, zi in enumerate(z):
            zi = float(zi)
            dh = c_km_s / (H_z(zi, *pr) * rd)
            dm = (c_km_s/rd) * quad(lambda w: 1.0/H_z(w, *pr), 0, zi,
                                    epsabs=1e-10, epsrel=1e-10, limit=200)[0]
            dv = (zi*(dm*rd)**2*(dh*rd))**(1/3) / rd
            out[i] = (dv, dm, dh)[typ[i]]
        return out
    print(f"\n(e) 双幂律(全精度参数)   :  χ²={chi2(theory_dpl(DPL)):9.2f}   [附录G表8: 5.50]")

    zs9 = [1.5, 2.0, 2.5, 3.0]
    def H_lcdm(zp, H0, Om):
        return H0 * np.sqrt(Om*(1+zp)**3 + (1-Om))
    print(f"\n(g) 附录G表9 复现 (D_H/r_d):")
    print(f"   {'z':>5} {'裸几何':>8} {'λ模型':>8} {'双幂律':>8} {'ΛCDM':>8}")
    for zq in zs9:
        bare = c_km_s/(H0_BARE*rd)/(1.0+zq)
        lam  = 1.1248 * bare
        dpl  = c_km_s/(H_z(zq, *DPL)*rd)
        lcdm = c_km_s/(H_lcdm(zq, 67.4, 0.315)*rd)
        print(f"   {zq:>5.2f} {bare:>8.2f} {lam:>8.2f} {dpl:>8.2f} {lcdm:>8.2f}")
    print("   [论文表9: 11.51/9.59/8.22/7.19 | 12.94/10.79/9.25/8.09"
          " | 12.90/10.02/8.04/6.63 | 12.77/9.98/8.03/6.63]")
except ImportError:
    print("\n(e)(g) 未装 scipy，跳过双幂律与表9复现")

# ---------------------------------------------------------------
# (f) 残差结构诊断 → 附录F: 11/13点 |r/σ|≤1.5; 两个例外
# ---------------------------------------------------------------
print(f"\n(f) 残差结构诊断 (λ=1.1248, 对角σ):")
sig = np.sqrt(np.diag(C))
d_all = (obs - 1.1248*t_bare)/sig
n_ok = 0
for i in range(n):
    flag = "OK" if abs(d_all[i]) <= 1.5 else "<-- 例外"
    if abs(d_all[i]) <= 1.5:
        n_ok += 1
    print(f"  #{i+1:2d} z={z[i]:6.3f} {OBS_NAME[typ[i]]:6s} obs={obs[i]:9.4f} "
          f"bare={t_bare[i]:9.4f} λ需={obs[i]/t_bare[i]:7.4f}  r/σ={d_all[i]:+7.2f}  {flag}")
print(f"  |r/σ|≤1.5 的点数: {n_ok}/13   [附录F: 11个]")
print(f"  预期例外: z=0.934 D_H (≈+4.5σ); z=2.33 D_H (≈−10.7σ, 与裸几何0.06σ一致)")
