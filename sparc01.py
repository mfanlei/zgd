#!/usr/bin/env python3
"""
SPARC 数据集 ZGD 旋转速度验证
==============================
读取 Lelli2016c 质量模型数据，使用推荐的质光比 0.7 计算重子总速度，
再按 ZGD 公式 v = (G M c² / Rₚ)^{1/4} 计算预测速度。
本脚本仅依赖 numpy, pandas, matplotlib，与数据文件放在同一目录即可运行。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==================== 物理常数 ====================
G  = 6.67430e-11          # m³ kg⁻¹ s⁻²
C  = 2.99792458e8         # m/s
RP = 4.4e26               # m，粒子视界半径（论文一）

# 质光比（盘和核球均采用 0.7）
ML_DISK = 0.7
ML_BUL  = 0.7

# ==================== 数据读取 ====================
# 注意文件名为 MassModels_Lelli2016c.mrt.txt
datafile = 'MassModels_Lelli2016c.mrt.txt'
if not os.path.exists(datafile):
    raise FileNotFoundError(f"找不到数据文件 {datafile}，请将其放在本脚本所在目录。")

# 数据文件为固定宽度格式，按照头文件说明定义列宽
colspecs = [(0, 11), (12, 18), (19, 25), (26, 32), (33, 38),
            (39, 45), (46, 52), (53, 59), (60, 67), (68, 76)]
colnames = ['ID', 'D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']

df = pd.read_fwf(datafile, colspecs=colspecs, names=colnames, skiprows=37)
# 跳过前面的说明行（实际行数可能需要根据文件调整，若报错可检查 skiprows）
# 如果读取失败，请检查文件开头的说明行数，调整 skiprows 值。

# 去除可能存在的空白 ID 行
df = df.dropna(subset=['ID'])

print(f"成功读取 {df.shape[0]} 条数据记录，共 {df['ID'].nunique()} 个星系。")

# ==================== 数据处理 ====================
# 计算恒星速度贡献（M/L=1 时的速度值需要乘以 sqrt(M/L)）
df['Vstar_disk'] = np.sqrt(ML_DISK) * df['Vdisk']
df['Vstar_bul']  = np.sqrt(ML_BUL)  * df['Vbul']

# 重子总速度贡献
df['Vbar'] = np.sqrt(df['Vgas']**2 + df['Vstar_disk']**2 + df['Vstar_bul']**2)

# 对每个星系，选取半径最大的若干点（平坦部分）
# 这里取每个星系半径最大的 5 个点
def select_outer_points(group, n_points=5):
    return group.nlargest(n_points, 'R')

outer_df = df.groupby('ID', group_keys=False).apply(select_outer_points).reset_index(drop=True)

print(f"筛选后得到 {outer_df.shape[0]} 条外围数据点。")

# 计算重子质量：从 v² = G M / r   =>  M = v² r / G
# 注意单位换算：R 是 kpc，V 是 km/s，需转为 SI
kpc_to_m = 3.085677581e19   # 1 kpc = 3.0857e19 m
km_to_m = 1000.0

outer_df['R_m'] = outer_df['R'] * kpc_to_m
outer_df['Vbar_ms'] = outer_df['Vbar'] * km_to_m
outer_df['Mbar'] = outer_df['Vbar_ms']**2 * outer_df['R_m'] / G  # kg

# ZGD 预测速度 (m/s)，再转回 km/s
outer_df['Vpred_ms'] = (G * outer_df['Mbar'] * C**2 / RP)**0.25
outer_df['Vpred'] = outer_df['Vpred_ms'] / km_to_m

# 比值
outer_df['ratio'] = outer_df['Vpred'] / outer_df['Vobs']

# 按星系聚合，取平均比值
gal_stats = outer_df.groupby('ID').agg(
    avg_ratio=('ratio', 'mean'),
    max_R=('R', 'max'),
    mean_Vobs=('Vobs', 'mean'),
    mean_Vpred=('Vpred', 'mean')
).reset_index()

# ==================== 统计结果 ====================
mean_ratio = gal_stats['avg_ratio'].mean()
median_ratio = gal_stats['avg_ratio'].median()
std_ratio = gal_stats['avg_ratio'].std()

print("\n====== SPARC 全样本 ZGD 验证 ======")
print(f"有效星系数：{len(gal_stats)}")
print(f"Vpred/Vobs 平均：{mean_ratio:.3f}")
print(f"Vpred/Vobs 中位：{median_ratio:.3f}")
print(f"Vpred/Vobs 标准差：{std_ratio:.3f}")
print(f"比值范围：{gal_stats['avg_ratio'].min():.3f} ~ {gal_stats['avg_ratio'].max():.3f}")

# 检查质量偏差方向
if mean_ratio > 1.0:
    mass_factor = mean_ratio**4
    print(f"平均 Vpred > Vobs (比值 {mean_ratio:.3f})，表明输入质量需除以 {mass_factor:.2f} 才能消除偏差。")
else:
    mass_factor = (1/mean_ratio)**4
    print(f"平均 Vpred < Vobs，输入质量需乘以 {mass_factor:.2f}。")

# ==================== 可视化 ====================
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 子图1：预测 vs 观测速度散点图
ax1 = axes[0]
ax1.scatter(gal_stats['mean_Vobs'], gal_stats['mean_Vpred'], alpha=0.5, edgecolors='k', linewidth=0.3)
ax1.plot([0, 300], [0, 300], 'r--', label='y = x')
ax1.set_xlabel('Observed V (km/s)')
ax1.set_ylabel('Predicted V (km/s)')
ax1.set_title(f'ZGD on SPARC (N={len(gal_stats)})\nMean ratio = {mean_ratio:.2f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：比值直方图
ax2 = axes[1]
ax2.hist(gal_stats['avg_ratio'], bins=40, edgecolor='k', alpha=0.7)
ax2.axvline(1.0, color='r', linestyle='--', label='Ratio = 1')
ax2.axvline(mean_ratio, color='b', linestyle='-', label=f'Mean = {mean_ratio:.2f}')
ax2.set_xlabel('Vpred / Vobs')
ax2.set_ylabel('Number of galaxies')
ax2.set_title('Distribution of velocity ratios')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sparc_zgd_validation.png', dpi=150)
plt.show()
print("\n图表已保存为 sparc_zgd_validation.png")

# ==================== 按中心密度（核球占比）分组 ====================
# 计算每个星系的平均核球速度贡献占比（外围点）
outer_df['bulge_frac'] = outer_df['Vbul']**2 / (outer_df['Vgas']**2 + outer_df['Vdisk']**2 + outer_df['Vbul']**2)
gal_bulge = outer_df.groupby('ID')['bulge_frac'].mean().reset_index()

# 将星系按核球占比分成三组
bins = [0, 0.05, 0.2, 1.0]
labels = ['Bulge-weak\n(bulge frac <0.05)', 'Moderate\n(0.05-0.2)', 'Bulge-strong\n(bulge frac >0.2)']
gal_bulge['bulge_group'] = pd.cut(gal_bulge['bulge_frac'], bins=bins, labels=labels, include_lowest=True)

# 合并到星系统计表
gal_stats = gal_stats.merge(gal_bulge[['ID', 'bulge_frac', 'bulge_group']], on='ID')

# 分组计算平均比值
group_stats = gal_stats.groupby('bulge_group', observed=True).agg(
    count=('avg_ratio', 'count'),
    mean_ratio=('avg_ratio', 'mean'),
    median_ratio=('avg_ratio', 'median'),
    std_ratio=('avg_ratio', 'std')
).reset_index()

print("\n====== 按核球占比分组统计 ======")
print(group_stats.to_string(index=False))

# 散点图：核球占比 vs 比值
fig2, ax = plt.subplots(figsize=(8, 5))
ax.scatter(gal_stats['bulge_frac'], gal_stats['avg_ratio'], alpha=0.5, edgecolors='k', linewidth=0.3)
ax.axhline(1.0, color='r', linestyle='--', label='Ratio = 1')
ax.set_xlabel('Mean Bulge Fraction (Vbul² / Vbar²)')
ax.set_ylabel('Vpred / Vobs')
ax.set_title('Central Density Proxy vs ZGD Prediction Offset')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sparc_bulge_fraction.png', dpi=150)
plt.show()
print("图表已保存为 sparc_bulge_fraction.png")