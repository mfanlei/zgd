#!/usr/bin/env python3
"""
验证 SPARC 的 C(q) 上升趋势是否稳健：
1. 同时计算三种 C 定义：C_ours, C_true, C_RAR
2. 对比外围 5 点 / 外围 3 点 / 全曲线
3. 看哪个定义下 C(q) 是上升，哪个是下降
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

G = 6.67430e-11
C_light = 2.99792458e8
RP = 4.4e26
A0 = C_light**2 / RP  # 2.043e-10 m/s^2

KPC = 3.085677581e19
KM = 1000.0
GAS_HE = 1.0
ML_DISK, ML_BUL = 0.5, 0.7

DF = pd.read_csv('MassModels_Lelli2016c.mrt.txt', sep=r'\s+', skiprows=25, header=None,
                 names=['ID','D','R','Vobs','e_Vobs','Vgas','Vdisk','Vbul','SBdisk','SBbul']).dropna(subset=['ID'])
assert DF['ID'].iloc[0] == 'CamB'
assert DF['ID'].nunique() == 175

vgas = np.sqrt(GAS_HE) * DF['Vgas']
vdisk = np.sqrt(ML_DISK) * DF['Vdisk']
vbul = np.sqrt(ML_BUL) * DF['Vbul']
DF['Vbar'] = np.sqrt(vgas**2 + vdisk**2 + vbul**2)

def nu_rar(x):
    x = np.maximum(x, 1e-10)
    return 1 / (1 - np.exp(-np.sqrt(x)))

def compute_definitions(sub):
    """对一组数据点计算三种 C 定义 + q"""
    R_m = sub['R'] * KPC
    Vb_ms = sub['Vbar'] * KM
    Vobs_ms = sub['Vobs'] * KM
    M_r = Vb_ms**2 * R_m / G
    V_geo2_ms = np.sqrt(G * M_r * C_light**2 / RP)  # 这是 v_geo^2
    
    # 定义 1：我们的 C = V_obs^2 / V_geo^2
    C_ours = Vobs_ms**2 / V_geo2_ms
    
    # 定义 2：第二期报告的 C_true = (V_obs^2 - V_bar^2) / V_geo^2
    C_true = (Vobs_ms**2 - Vb_ms**2) / V_geo2_ms
    
    # 定义 3：RAR 精确插值的 C
    g_bar = Vb_ms**2 / R_m
    x_bar = g_bar / A0
    g_obs_RAR = nu_rar(x_bar) * g_bar
    # C_RAR = (g_obs_RAR - g_bar) / g_geo = (nu-1)*g_bar / (sqrt(g_bar*a_0))
    g_geo = np.sqrt(g_bar * A0)
    C_RAR = (g_obs_RAR - g_bar) / g_geo
    
    # q 定义
    q = Vb_ms**2 / V_geo2_ms
    
    return pd.DataFrame({
        'C_ours': C_ours,
        'C_true': C_true,
        'C_RAR': C_RAR,
        'q': q
    })

def aggregate_by_galaxy(mode, n_points=None):
    """mode: 'outer5' / 'outer3' / 'all'"""
    if mode == 'all':
        sub = DF.copy()
    else:
        n = 5 if mode == 'outer5' else 3
        sub = (DF.sort_values(['ID','R'], ascending=[True, False])
                 .groupby('ID', as_index=False).head(n).copy().reset_index(drop=True))
    
    defs = compute_definitions(sub)
    sub = pd.concat([sub[['ID']].reset_index(drop=True), defs], axis=1)
    
    # 聚合到星系级（中位数）
    gal = sub.groupby('ID', as_index=False).agg(
        q=('q', 'median'),
        C_ours=('C_ours', 'median'),
        C_true=('C_true', 'median'),
        C_RAR=('C_RAR', 'median')
    )
    return gal.dropna()

def binning_and_slope(gal, label):
    bins = np.linspace(0, 0.5, 6)
    gal['q_bin'] = pd.cut(gal['q'], bins)
    
    print(f"\n{'='*70}")
    print(f"模式：{label}，星系总数：{len(gal)}")
    print(f"{'='*70}")
    print(f"{'q_bin':<15}{'N':<6}{'C_ours':<12}{'C_true':<12}{'C_RAR':<12}")
    print("-"*70)
    
    binned = gal.groupby('q_bin', observed=True).agg(
        N=('q', 'count'),
        C_ours=('C_ours', 'median'),
        C_true=('C_true', 'median'),
        C_RAR=('C_RAR', 'median')
    )
    for q_bin, row in binned.iterrows():
        if row['N'] < 3: continue
        print(f"{str(q_bin):<15}{int(row['N']):<6}{row['C_ours']:<12.3f}{row['C_true']:<12.3f}{row['C_RAR']:<12.3f}")
    
    # 线性回归斜率
    from scipy.stats import linregress
    print("-"*70)
    for col in ['C_ours', 'C_true', 'C_RAR']:
        sl, ic, rv, pv, se = linregress(gal['q'], gal[col])
        print(f"  {col:<10} 斜率 = {sl:+.3f} ± {se:.3f}  (R² = {rv**2:.3f})")
    return gal

# 三种采样模式
print("="*70)
print("C(q) 的三定义对比：C_ours vs C_true vs C_RAR")
print("="*70)

gal_outer5 = aggregate_by_galaxy('outer5')
gal_outer3 = aggregate_by_galaxy('outer3')
gal_all = aggregate_by_galaxy('all')

binning_and_slope(gal_outer5, '外围 5 点')
binning_and_slope(gal_outer3, '外围 3 点')
binning_and_slope(gal_all, '全曲线（所有点中位数）')

# 额外诊断：对 C_true 和 C_RAR 分别做深端统计
print("\n" + "="*70)
print("深端（q < 0.2）统计")
print("="*70)
for label, gal in [('外5', gal_outer5), ('外3', gal_outer3), ('全部', gal_all)]:
    deep = gal[gal['q'] < 0.2]
    print(f"{label}: N={len(deep)}, C_ours中位={deep['C_ours'].median():.3f}, "
          f"C_true中位={deep['C_true'].median():.3f}, C_RAR中位={deep['C_RAR'].median():.3f}")

# 绘图 (使用英文标签避免乱码)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 可选：让字体风格更接近 LaTeX 书体
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.scatter(gal_outer5['q'], gal_outer5['C_ours'], alpha=0.4,
           label=r'$C_{\mathrm{ours}} = V_{\mathrm{obs}}^2/V_{\mathrm{geo}}^2$', s=15)
ax.scatter(gal_outer5['q'], gal_outer5['C_true'], alpha=0.4,
           label=r'$g = (V_{\mathrm{obs}}^2 - V_{\mathrm{bar}}^2)/V_{\mathrm{geo}}^2$', s=15)
ax.scatter(gal_outer5['q'], gal_outer5['C_RAR'], alpha=0.4,
           label='RAR interpolation', s=15)
ax.set_xlabel(r'$q = V_{\mathrm{bar}}^2/V_{\mathrm{geo}}^2$')
ax.set_ylabel('Coefficient value')
ax.set_title('Outer 5 points: Comparison of three definitions')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 分箱图
ax = axes[1]
bins = np.linspace(0, 0.5, 6)
gal_outer5['q_bin'] = pd.cut(gal_outer5['q'], bins)
binned = gal_outer5.groupby('q_bin', observed=True).agg(
    q_med=('q', 'median'),
    C_ours=('C_ours', 'median'),
    C_true=('C_true', 'median'),
    C_RAR=('C_RAR', 'median')
)
ax.plot(binned['q_med'], binned['C_ours'], 'o-',
        label=r'$C_{\mathrm{ours}}$', color='tab:blue')
ax.plot(binned['q_med'], binned['C_true'], 's-',
        label=r'Pure geometric scale $g$', color='tab:orange')
ax.plot(binned['q_med'], binned['C_RAR'], '^-',
        label='RAR interpolation', color='tab:green')
ax.axhline(0.74, color='r', ls='--', alpha=0.5,
           label=r'$C_{\mathrm{deep}} = 0.74$')
ax.set_xlabel(r'$q$')
ax.set_ylabel('Coefficient (binned median)')
ax.set_title('Binned median: which definition rises?')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('c_q_verification.png', dpi=200)
plt.close()
print("\n图已保存：c_q_verification.png (图内文字已统一为英文，无乱码)")

# ==================== 追加：环境梯度显著性的 Mann-Whitney 检验 ====================
from scipy.stats import mannwhitneyu

print("\n" + "="*70)
print("【Mann-Whitney 检验：q<0.1 深端矮星系 vs q>0.1 盘星系】")
print("="*70)

gal = gal_outer5.copy()
deep = gal[gal['q'] < 0.1]['C_true']  # 深端 g 值
main = gal[gal['q'] > 0.1]['C_true']  # 盘星系 g 值

print(f"深端 (q<0.1) N={len(deep)}, 中位 g={deep.median():.3f}")
print(f"主样本 (q>0.1) N={len(main)}, 中位 g={main.median():.3f}")

stat, p = mannwhitneyu(deep, main, alternative='two-sided')
print(f"Mann-Whitney U={stat:.1f}, p-value={p:.3e}")
if p < 0.05:
    print("结论：存在统计学显著差异（p < 0.05）。")
else:
    print("结论：差异不显著（p > 0.05），只能称为提示性趋势。")
print("="*70)
