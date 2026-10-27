#!/usr/bin/env python3
"""
SPARC ZGD 旋转速度验证 — 最终版 (渐近极限与上确界检验)
=========================================================
检验目标：
1. 渐近公式 v = (G*M*c^2/R_p)^{1/4} 的整体偏移与等效 a0。
2. 点质量近似的额外高估断层（L型）。
3. 盘主导 vs 中心集中的二元对比。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.stats import mannwhitneyu, wilcoxon

# ==================== 常数与配置 ====================
G, C, RP = 6.67430e-11, 2.99792458e8, 4.4e26
KPC, KM = 3.085677581e19, 1000.0
GAS_HE_FACTOR = 1.0
ML_DISK, ML_BUL = 0.5, 0.7

# ==================== 数据读取 ====================
datafile = 'MassModels_Lelli2016c.mrt.txt'
if not os.path.exists(datafile):
    raise FileNotFoundError(datafile)

DF = pd.read_csv(datafile, sep=r'\s+', skiprows=25, header=None,
                 names=['ID','D','R','Vobs','e_Vobs','Vgas','Vdisk','Vbul','SBdisk','SBbul'])
DF = DF.dropna(subset=['ID'])

# 加上断言，确保解析无误
assert DF['ID'].iloc[0] == 'CamB', "第一行不是 CamB，表头行数不对！"
assert DF['ID'].nunique() == 175, f"星系数不对，只有 {DF['ID'].nunique()} 个！"

print("="*80)
print("SPARC ZGD 最终验证 — 渐近极限与上确界检验")
print("="*80)
print(f"数据：{DF.shape[0]} 条记录，{DF['ID'].nunique()} 个星系\n")

# ==================== 核心计算 ====================
def compute_gal_stats(n_points=5):
    vgas = np.sqrt(GAS_HE_FACTOR) * DF['Vgas']
    vdisk = np.sqrt(ML_DISK) * DF['Vdisk']
    vbul  = np.sqrt(ML_BUL)  * DF['Vbul']
    vbar = np.sqrt(vgas**2 + vdisk**2 + vbul**2)

    d = DF.copy()
    d['Vbar'] = vbar
    # 取外围 N 个点作为平坦区
    outer = (d.sort_values(['ID','R'], ascending=[True, False])
               .groupby('ID', as_index=False).head(n_points).copy())

    outer['R_m']   = outer['R'] * KPC
    outer['Vb_ms'] = outer['Vbar'] * KM
    outer['M_r']   = outer['Vb_ms']**2 * outer['R_m'] / G  # 包围质量
    max_m = outer.groupby('ID')['M_r'].transform('max')
    outer['M_tot'] = max_m

    # 渐近公式 V = (GMc^2/Rp)^0.25
    outer['Vpred_zdg'] = ((G * outer['M_r'] * C**2 / RP)**0.25) / KM
    outer['ratio_zdg'] = outer['Vpred_zdg'] / outer['Vobs']
    
    outer['Vpred_pt'] = ((G * outer['M_tot'] * C**2 / RP)**0.25) / KM
    outer['ratio_pt'] = outer['Vpred_pt'] / outer['Vobs']

    m_gas  = vgas.loc[outer.index].values**2
    m_disk = ML_DISK * DF.loc[outer.index, 'Vdisk'].values**2
    m_bul  = ML_BUL  * DF.loc[outer.index, 'Vbul'].values**2
    m_tot_arr = np.where((m_gas+m_disk+m_bul) > 0, m_gas+m_disk+m_bul, np.nan)
    outer['bulge_frac'] = m_bul / m_tot_arr

    gal = outer.groupby('ID', as_index=False).agg(
        avg_ratio_zdg=('ratio_zdg', 'mean'),
        avg_ratio_pt=('ratio_pt', 'mean'),
        bulge_frac=('bulge_frac', 'mean'),
        mean_Vobs=('Vobs', 'mean'),
        mean_Vpred_zdg=('Vpred_zdg', 'mean'),
    )
    gal['delta'] = gal['avg_ratio_pt'] - gal['avg_ratio_zdg']
    return gal.dropna(subset=['avg_ratio_zdg', 'bulge_frac'])

def boot_ci(x, stat='median', n=3000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x); x = x[~np.isnan(x)]
    if len(x) < 3: return (np.nan,)*3
    f = np.median if stat == 'median' else np.mean
    vals = [f(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return f(x), np.percentile(vals, 16), np.percentile(vals, 84)

gal = compute_gal_stats(5)
print(f"有效星系：{len(gal)}")

# ==================== 整体统计 ====================
print("-" * 80)
print(f"{'指标':<15} {'中位数':<10} {'68% CI':<24} {'均值':<10}")
print("-" * 80)
for key in ['avg_ratio_zdg', 'avg_ratio_pt', 'delta']:
    m, ml, mh = boot_ci(gal[key], 'median')
    a, al, ah = boot_ci(gal[key], 'mean')
    print(f"{key:<15} {m:.4f}     [{ml:.4f},{mh:.4f}]     {a:.4f}")

# ==================== 上界定理检验 ====================
violations = (gal['avg_ratio_pt'] < 1).sum()
print(f"\n上界定理检验：违反 v_obs <= v_pt 的星系数 = {violations} / {len(gal)}")
if violations > 0:
    w_stat, w_p = wilcoxon(gal['avg_ratio_pt'] - 1, alternative='greater')
    print(f"Wilcoxon 符号秩检验：p = {w_p:.2e}")

# ==================== 分组与二元对比 ====================
bins = [0, 0.05, 0.2, 1.0]
labels = ['weak (<0.05)', 'moderate (0.05-0.2)', 'strong (>0.2)']
gal['group'] = pd.cut(gal['bulge_frac'], bins=bins, labels=labels, include_lowest=True)

print("\n" + "-" * 80)
print(f"{'组别':<20} {'n':<5} {'中位_zdg':<10} {'中位_pt':<10} {'delta':<10}")
print("-" * 80)
for lab in labels:
    sub = gal[gal['group'] == lab]
    if len(sub) < 3: continue
    print(f"{lab:<20} {len(sub):<5} {sub['avg_ratio_zdg'].median():.4f}     "
          f"{sub['avg_ratio_pt'].median():.4f}     {sub['delta'].median():.4f}")

print("\n二元对比：盘主导 (f_bul < 0.05) vs 中心集中 (f_bul >= 0.05)")
d_grp = gal[gal['bulge_frac'] < 0.05]
c_grp = gal[gal['bulge_frac'] >= 0.05]

for key in ['avg_ratio_zdg', 'delta']:
    m_d = d_grp[key].median()
    m_c = c_grp[key].median()
    delta_med = m_c - m_d
    rng = np.random.default_rng(42)
    diffs = [np.median(rng.choice(c_grp[key], len(c_grp), True)) -
             np.median(rng.choice(d_grp[key], len(d_grp), True))
             for _ in range(5000)]
    dl, dh = np.percentile(diffs, [16, 84])
    p = mannwhitneyu(c_grp[key], d_grp[key], alternative='two-sided').pvalue
    print(f"  {key:<15} 盘: {m_d:.4f}  集: {m_c:.4f}  Δ = {delta_med:+.4f} "
          f"[{dl:+.4f},{dh:+.4f}]  p={p:.2e}")

# ==================== 作图 ====================
plt.rcParams.update({'font.size': 11})

# 图 1：散点与直方图
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
ax = axes[0]
ax.scatter(gal['mean_Vobs'], gal['mean_Vpred_zdg'], s=18, alpha=0.55, edgecolors='k', linewidth=0.3)
ax.plot([0, 300], [0, 300], 'r--', lw=1.2, label='y = x')
ax.set_xlabel('$V_{obs}$ (km/s)')
ax.set_ylabel('$V_{pred}$ (km/s)')
ax.set_title(f'ZGD Asymptotic Formula (N={len(gal)})\nMedian ratio = {gal["avg_ratio_zdg"].median():.3f}')
ax.legend(frameon=False); ax.grid(alpha=0.25)

ax = axes[1]
ax.hist(gal['avg_ratio_zdg'], bins=40, edgecolor='k', alpha=0.7, linewidth=0.5)
ax.axvline(1.0, color='r', ls='--', lw=1.2, label='ratio = 1')
ax.axvline(gal['avg_ratio_zdg'].median(), color='b', ls='-', lw=1.2, label=f'Median = {gal["avg_ratio_zdg"].median():.3f}')
ax.axvline(gal['avg_ratio_zdg'].mean(), color='g', ls=':', lw=1.2, label=f'Mean = {gal["avg_ratio_zdg"].mean():.3f}')
ax.set_xlabel('$V_{pred}/V_{obs}$')
ax.set_ylabel('Number of galaxies')
ax.set_title('Velocity ratio distribution')
ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig('sparc_zgd_validation.png', dpi=200); plt.close()

# 图 2：L 型断层
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.scatter(gal['bulge_frac'], gal['delta'], s=20, alpha=0.6, edgecolors='k', linewidth=0.3)
ax.axhline(0, color='k', ls='--', lw=1.0)
ax.axvline(0.05, color='k', ls='-.', lw=1.0, alpha=0.7, label='binary split t=0.05')
ax.set_xlabel('Bulge mass fraction')
ax.set_ylabel('$\\Delta$ ratio')
ax.set_title('Point-mass overestimation as a function of central density')
ax.legend(frameon=False); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig('sparc_delta_overestimation.png', dpi=200); plt.close()

print("\n图已保存：sparc_zgd_validation.png, sparc_delta_overestimation.png")
