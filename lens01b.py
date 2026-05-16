#!/usr/bin/env python3
"""
ZGD 引力透镜验证 B（ZGD 完整公式 + 有效质光比 0.51）
========================================================
使用 ZGD 偏折角公式（牛顿项 + 几何修正项）与
SPARC 校准的有效质光比 0.51，
计算 15 个透镜系统的理论偏折角并与观测对比。
验证逻辑：若几何修正项在透镜碰撞参数处被点质量近似严重高估，
          则 ZGD 公式应系统性高于观测——与 SPARC 的偏移同源。
"""

import numpy as np

# ==================== 物理常数 ====================
G  = 6.67430e-11
C  = 2.99792458e8
RP = 4.4e26                # 粒子视界半径
MSUN = 1.98847e30
KPC = 3.085677581e19       # m / kpc

UPS_EFF  = 0.51            # SPARC 强核球组校准的有效质光比
UPS_OLD  = 0.7              # 原 SPARC 基准质光比

# ==================== 判定阈值 ====================
RATIO_LO = 0.80             # 中位比值下限
RATIO_HI = 1.20             # 中位比值上限
# 注：若 ZGD 点质量公式在透镜内区严重高估几何修正项，
#     预期中位比值将显著高于 1.20。

# ==================== 透镜数据 ====================
lens_data = [
    ("SDSS J0946+1006 (内环)", 2.5e11, 4.8, 1.43),
    ("SDSS J0946+1006 (外环)", 2.5e11, 4.8, 2.07),
    ("QSO 2237+0305",          3.0e11, 0.70, 0.90),
    ("Abell 1689",             2.0e14, 150.0, 45.0),
    ("B1608+656",              1.0e11, 3.0, 1.14),
    ("MG1549+3047",            2.0e10, 2.0, 0.85),
    ("B0218+357",              5.0e9,  1.0, 0.17),
    ("MG2016+112",             3.0e11, 5.0, 1.76),
    ("RXJ0911+0551",           1.5e11, 3.5, 1.24),
    ("Q0142-100",              8.0e10, 2.5, 1.12),
    ("B1600+434",              6.0e10, 2.0, 0.70),
    ("PKS1830-211",            4.0e10, 1.8, 0.50),
    ("B0712+472",              5.0e10, 2.0, 0.73),
    ("SDSS J0037-0942",        1.8e11, 3.5, 1.18),
    ("SDSS J0956+5100",        2.2e11, 4.0, 1.38),
]

print("ZGD 引力透镜验证 B（ZGD 完整公式 + 有效质光比 0.51）")
print(f"{'系统':<25} {'b(kpc)':<8} {'α_N':<8} {'α_geo':<8} {'α_tot':<8} {'θ_obs':<8} {'比值':<8}")
print("-" * 75)

ratios = []
for name, M_old, b_kpc, theta_obs in lens_data:
    M_true = M_old * (UPS_EFF / UPS_OLD)
    b_m = b_kpc * KPC
    M_kg = M_true * MSUN

    alpha_N_rad = 4 * G * M_kg / (C**2 * b_m)
    alpha_N = alpha_N_rad * 206265

    alpha_geo_rad = 2 * np.pi / C * np.sqrt(G * M_kg / RP)
    alpha_geo = alpha_geo_rad * 206265

    alpha_tot = alpha_N + alpha_geo
    ratio = alpha_tot / theta_obs
    ratios.append(ratio)
    print(f"{name:<25} {b_kpc:<8.1f} {alpha_N:<8.3f} {alpha_geo:<8.3f} {alpha_tot:<8.3f} {theta_obs:<8.2f} {ratio:<8.3f}")

med = np.median(ratios)
avg = np.mean(ratios)

print("-" * 75)
print(f"有效透镜系统数: {len(ratios)}")
print(f"中位比值: {med:.3f}")
print(f"平均比值: {avg:.3f}")

# ==================== 判定 ====================
if RATIO_LO <= med <= RATIO_HI:
    print(f"判定: 中位比值 {med:.3f} 落在 [{RATIO_LO}, {RATIO_HI}] 内。")
    print("       ZGD 完整公式在透镜系统中与观测基本一致。")
    print("       几何修正项的贡献在透镜碰撞参数处不可忽略。")
elif med < RATIO_LO:
    print(f"判定: 中位比值 {med:.3f} 低于 {RATIO_LO}。")
    print("       ZGD 完整公式系统性低估观测偏折角。")
else:  # med > RATIO_HI
    print(f"判定: 中位比值 {med:.3f} 显著高于 {RATIO_HI}。")
    print("       ZGD 点质量公式在透镜内区系统性高估几何修正项。")
    print("       这与 SPARC 整体偏移在物理上同源——")
    print("       点质量近似在透镜碰撞参数处（b~1-5 kpc）的高估效应")
    print("       比在旋转曲线外围（r~2-5 R_d）更为严重。")