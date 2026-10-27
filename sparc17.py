#!/usr/bin/env python3
"""
SPARC 收口：leave-bin-out回归、bin4/5构成诊断、full-RAR曲线、174点名
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

G, C_LIGHT, RP = 6.67430e-11, 2.99792458e8, 4.4e26
KPC, KM = 3.085677581e19, 1000.0
GAS_HE_FACTOR = 1.0
ML_DISK, ML_BUL = 0.5, 0.7
A0_ZGD = C_LIGHT**2 / RP
A_B_RAR = 1.20e-10

DF = pd.read_csv('MassModels_Lelli2016c.mrt.txt', sep=r'\s+', skiprows=25, header=None,
                 names=['ID','D','R','Vobs','e_Vobs','Vgas','Vdisk','Vbul','SBdisk','SBbul']).dropna(subset=['ID'])

# 加上断言，确保解析无误
assert DF['ID'].iloc[0] == 'CamB', "第一行不是 CamB，表头行数不对！"
assert DF['ID'].nunique() == 175, f"星系数不对，只有 {DF['ID'].nunique()} 个！"

vgas = np.sqrt(GAS_HE_FACTOR) * DF['Vgas']
vdisk = np.sqrt(ML_DISK) * DF['Vdisk']
vbul  = np.sqrt(ML_BUL)  * DF['Vbul']
DF['Vbar'] = np.sqrt(vgas**2 + vdisk**2 + vbul**2)

flat_df = DF.groupby('ID').apply(lambda s: pd.Series({
    'V_in': s.nsmallest(3,'R')['Vobs'].mean(),
    'V_out': s.nlargest(3,'R')['Vobs'].mean(),
    'V_max': s['Vobs'].max(),
    'R_max': s['R'].max(),
}), include_groups=False).reset_index()
flat_df['flatness'] = flat_df['V_out'] / flat_df['V_in']

vflat = DF.groupby('ID').apply(lambda s: s.nlargest(3,'R')['Vobs'].median(),
                               include_groups=False).rename('V_flat').reset_index()

outer = (DF.sort_values(['ID','R'], ascending=[True, False])
           .groupby('ID', as_index=False).head(5).copy())
outer['R_m'] = outer['R'] * KPC
outer['Vb_ms'] = outer['Vbar'] * KM
outer['Vobs_ms'] = outer['Vobs'] * KM
outer['M_r'] = outer['Vb_ms']**2 * outer['R_m'] / G
outer['V_geo2_ms'] = np.sqrt(G * outer['M_r'] * C_LIGHT**2 / RP)
outer['q'] = (outer['Vb_ms']**2) / outer['V_geo2_ms']
outer['g'] = (outer['Vobs_ms']**2 - outer['Vb_ms']**2) / outer['V_geo2_ms']
outer['C_calc'] = outer['g'] + outer['q']

outer['M_tot'] = outer.groupby('ID')['M_r'].transform('max')
outer['V_geo_pt2_ms'] = (G * outer['M_tot'] * C_LIGHT**2 / RP)**0.5
outer['eta'] = (outer['Vobs_ms']**2 - outer['Vb_ms']**2) / outer['V_geo_pt2_ms']

def build_gal(o):
    return o.groupby('ID', as_index=False).agg(
        q=('q','median'), g=('g','median'), C=('C_calc','median'),
        eta=('eta','median'),
        Vobs_med=('Vobs','median'), Vbar_med=('Vbar','median'),
        Vgas_med=('Vgas','median'), Vbul_med=('Vbul','median'),
    ).dropna()

gal_pos = build_gal(outer[outer['g'] > 0])
gal_all = build_gal(outer) 

print(f"g>0 样本: N={len(gal_pos)}   全样本: N={len(gal_all)}")

bins = np.linspace(0, 0.5, 6)
gal_pos['q_bin'] = pd.cut(gal_pos['q'], bins)
gal_pos = gal_pos.merge(flat_df[['ID','flatness','V_max']], on='ID')
gal_flat = gal_pos[gal_pos['flatness'] > 1.3].copy()

def reg(g, label):
    if len(g) < 5: return
    sl, ic, r, p, se = linregress(g['q'], g['C'])
    print(f"  {label:<35} N={len(g):<4}  slope = {sl:+.3f} ± {se:.3f}")

print("\n[leave-bin-out 稳健性]")
reg(gal_pos, "full g>0 sample")
reg(gal_flat, "flat subsample (flatness>1.3)")
g_no_bin5 = gal_pos[gal_pos['q'] < 0.4]
reg(g_no_bin5, "exclude bin5 (q>0.4)")
g_no_bin45 = gal_pos[gal_pos['q'] < 0.3]
reg(g_no_bin45, "exclude bin4+5 (q>0.3)")
g_no_bin4_only = gal_pos[(gal_pos['q'] < 0.3) | ((gal_pos['q'] > 0.4) & (gal_pos['q'] <= 0.5))]
reg(g_no_bin4_only, "exclude bin4 only (0.3<q<0.4)")

print("\n[bin4/5 构成诊断（q>0.3 vs q<0.3）]")
gal_pos['is_hi_q'] = gal_pos['q'] > 0.3
for col in ['Vobs_med', 'Vbar_med', 'Vgas_med', 'Vbul_med', 'V_max', 'flatness']:
    if col not in gal_pos.columns: continue
    hi = gal_pos[gal_pos['is_hi_q']][col].median()
    lo = gal_pos[~gal_pos['is_hi_q']][col].median()
    if lo == 0:
        ratio_str = "inf"
    else:
        ratio_str = f"{hi/lo:.2f}"
    print(f"  {col:<12} high-q: {hi:.2f}   low-q: {lo:.2f}   ratio = {ratio_str}")

print(f"\n[bin4/5 计数]")
print(f"  bin4 (0.3<q<0.4): {((gal_pos['q']>=0.3)&(gal_pos['q']<0.4)).sum()}")
print(f"  bin5 (0.4<q<0.5): {((gal_pos['q']>=0.4)&(gal_pos['q']<0.5)).sum()}")

print(f"\n[η 与 C 的样本一致性]")
print(f"  g>0 裁剪后 η 中位 = {gal_pos['eta'].median():.3f}, N={len(gal_pos)}")
print(f"  g>0 裁剪后 η>1 违反数 = {(gal_pos['eta']>1).sum()}")
print(f"  g>0 裁剪后 C 中位 = {gal_pos['C'].median():.3f}")
print(f"  g>0 裁剪后 C 的 IQR = [{gal_pos['C'].quantile(0.25):.3f}, {gal_pos['C'].quantile(0.75):.3f}]")

print("\n[Vizier 175 名单]")
try:
    from astroquery.vizier import Vizier
    cats = Vizier.get_catalogs('J/AJ/152/156')
    official = None
    for c in cats:
        if 'Name' in c.colnames:
            official = set(c['Name'])
            break
    if official:
        mine = set(DF['ID'].unique())
        print(f"  官方: {len(official)}, 本地: {len(mine)}")
        print(f"  缺失: {sorted(official - mine)}")
        print(f"  多余: {sorted(mine - official)}")
    else:
        print("  未在 Vizier 目录中找到 'Name' 列")
except Exception as e:
    print(f"  astroquery 不可用或查询失败: {e}")
    print(f"  本地星系数: {len(DF['ID'].unique())}")

def nu_standard(x):
    x = np.asarray(x, dtype=float)
    return 1/(1 - np.exp(-np.sqrt(np.maximum(x, 1e-10))))

def C_full_RAR(q):
    x = q**2 * A0_ZGD / A_B_RAR
    return q * nu_standard(x)

q_grid = np.linspace(0.01, 0.5, 200)
C_pure = np.full_like(q_grid, np.sqrt(A_B_RAR / A0_ZGD))
C_rar_full = C_full_RAR(q_grid)
C_add = q_grid + 0.593

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, gal, title in [(axes[0], gal_flat, f'A: flat subsample (N={len(gal_flat)})'),
                        (axes[1], gal_pos, f'B: full g>0 sample (N={len(gal_pos)})')]:
    binned = gal.copy()
    binned['qb'] = pd.cut(binned['q'], bins)
    rows = []
    for b, sub in binned.groupby('qb', observed=True):
        if len(sub) < 3: continue
        rng = np.random.default_rng(42)
        meds = [np.median(rng.choice(sub['C'].values, len(sub), True)) for _ in range(2000)]
        rows.append(dict(q=b.mid, C=sub['C'].median(),
                         lo=np.percentile(meds,16), hi=np.percentile(meds,84), n=len(sub)))
    bb = pd.DataFrame(rows)
    ax.errorbar(bb['q'], bb['C'], yerr=[bb['C']-bb['lo'], bb['hi']-bb['C']],
                fmt='o', capsize=4, color='tab:blue', label='Observed C (bootstrap 68%)')
    ax.plot(q_grid, C_pure, 'b--', lw=1.8, label=f'Pure sqrt RAR (a_B=1.20)')
    ax.plot(q_grid, C_rar_full, 'b-', lw=2, label='Full RAR (standard-ν)')
    ax.plot(q_grid, C_add, 'g:', lw=2, label='Additive ansatz (g=0.593)')
    ax.axhline(0.740, color='r', lw=1.5, alpha=0.6, label='Deep-MOND C ≈ 0.74')
    ax.set_xlabel('q'); ax.set_ylabel('C = V_obs²/V_geo²')
    ax.set_title(title); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('sparc_C_plane_final.png', dpi=200); plt.close()
print("\n图已保存：sparc_C_plane_final.png")

# ==================== 最终输出确认 ====================
print("\n" + "="*50)
print("【最终数据确认】")
print(f"q 中位数: {gal_pos['q'].median():.3f}")
print(f"g 中位数: {gal_pos['g'].median():.3f}")

# 计算两端论证的锚点 (深端平台和加性模型拟合值)
# 这里直接拟合加性模型 C = q + g，得出 g 的拟合值
from scipy.optimize import curve_fit
def linear_model(q, g_val): return q + g_val
popt, _ = curve_fit(linear_model, gal_pos['q'], gal_pos['C'])
print(f"加性 ansatz 拟合 g 值: {popt[0]:.3f}")

# 计算 C_deep (q < 0.2 的中位数)
C_deep = gal_pos[gal_pos['q'] < 0.2]['C'].median()
C_deep_lo = gal_pos[gal_pos['q'] < 0.2]['C'].quantile(0.16)
C_deep_hi = gal_pos[gal_pos['q'] < 0.2]['C'].quantile(0.84)
print(f"C_deep (q<0.2 中位数): {C_deep:.3f} [{C_deep_lo:.3f}, {C_deep_hi:.3f}]")
print("="*50)

# ==================== 补充硬伤验证 ====================
print("\n" + "="*50)
print("【补充验证】")
# 1. bin1 (q<0.1) 的精确中位数与标准差
bin1 = gal_pos[(gal_pos['q'] >= 0.0) & (gal_pos['q'] < 0.1)]['C']
print(f"bin1 (q<0.1) 中位数: {bin1.median():.3f}, 标准差: {bin1.std():.3f}, N={len(bin1)}")

# 2. C_deep 的精确 CI
c_deep_sub = gal_pos[gal_pos['q'] < 0.2]['C']
print(f"C_deep 中位数: {c_deep_sub.median():.3f}, IQR: [{c_deep_sub.quantile(0.16):.3f}, {c_deep_sub.quantile(0.84):.3f}]")

# 3. g>1 的确切数量
print(f"g>1 的星系数: {(gal_pos['g'] > 1).sum()}")
