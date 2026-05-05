#!/usr/bin/env python3
"""
ZGD 理论验证：近场 H0-物质密度对比
基于 Cosmicflows-4 table3.dat 生成局部 H0 全天地图与密度图，检验相关性。

环境要求：numpy, scipy, healpy, matplotlib
"""

import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy import stats

# ===================== 配置 =====================
TABLE3_PATH = "table3.dat"          # 输入文件（与代码同目录）
NSIDE = 32
MIN_GROUPS_PER_PIX = 5               # 像素最小星系群数
H0_MIN, H0_MAX = 20.0, 200.0       # H0 异常值剔除范围

# 输出文件名
H0_MAP_FILE = "h0_table3_n32.fits"
DENSITY_MAP_FILE = "density_table3_n32.fits"
MASK_MAP_FILE = "h0_mask_table3_n32.fits"
SCATTER_PLOT_FILE = "h0_vs_density_scatter.png"

# ===================== 步骤 1：加载 table3.dat =====================
print("=" * 60)
print("步骤 1：加载 table3.dat（固定宽度解析）")
print("=" * 60)

records = []
with open(TABLE3_PATH, "r") as f:
    for line_num, line in enumerate(f, 1):
        line = line.rstrip(os.linesep)
        if len(line) < 87:
            continue
        try:
            # 按字节位置切片（注意：Python 切片 end 是不包含的）
            pgc     = line[0:7].strip()
            dmzp    = line[8:14].strip()
            dmav    = line[15:21].strip()
            e_dmav  = line[22:27].strip()
            vcmb    = line[28:33].strip()
            ra_deg  = line[34:42].strip()
            de_deg  = line[43:51].strip()
            glon    = line[52:60].strip()
            glat    = line[61:69].strip()
            sgl     = line[70:78].strip()
            sgb     = line[79:87].strip()

            # 跳过 DMav 为 0 或空白的情况
            if not dmav or dmav == "0.000" or float(dmav) <= 0:
                continue
            if not e_dmav or float(e_dmav) <= 0:
                continue
            if not vcmb or not vcmb.strip():
                continue

            dmav_f   = float(dmav)
            e_dmav_f = float(e_dmav)
            vcmb_f   = float(vcmb)
            glon_f   = float(glon)
            glat_f   = float(glat)

            records.append({
                "pgc": pgc,
                "DMav": dmav_f,
                "e_DMav": e_dmav_f,
                "Vcmb": vcmb_f,
                "GLON": glon_f,
                "GLAT": glat_f,
            })
        except Exception as e:
            # 静默跳过解析失败的行
            continue

n_valid = len(records)
print(f"有效星系群数量: {n_valid}")

# ===================== 步骤 2：距离换算 =====================
print("\n" + "=" * 60)
print("步骤 2：距离模数 -> 物理距离")
print("=" * 60)

DMav_arr   = np.array([r["DMav"] for r in records])
e_DMav_arr = np.array([r["e_DMav"] for r in records])
Vcmb_arr   = np.array([r["Vcmb"] for r in records])
GLON_arr   = np.array([r["GLON"] for r in records])
GLAT_arr   = np.array([r["GLAT"] for r in records])

# d = 10^((DMav - 25) / 5)  [Mpc]
d_arr = 10.0 ** ((DMav_arr - 25.0) / 5.0)
# sigma_d = d * ln(10)/5 * e_DMav ~ 0.460517 * d * e_DMav
sigma_d_arr = d_arr * (np.log(10.0) / 5.0) * e_DMav_arr

print(f"距离范围: {d_arr.min():.3f} - {d_arr.max():.3f} Mpc")
print(f"距离中位数: {np.median(d_arr):.3f} Mpc")

# ===================== 步骤 3：坐标映射到 HEALPix =====================
print("\n" + "=" * 60)
print("步骤 3：坐标映射（Galactic -> HEALPix RING, Nside=32）")
print("=" * 60)

npix = hp.nside2npix(NSIDE)
print(f"总像素数: {npix}")

# healpy.ang2pix 要求 theta=colatitude=90-lat, phi=longitude (radians)
# 但 lonlat=True 时直接传 (lon, lat) in degrees
pix_indices = hp.ang2pix(NSIDE, GLON_arr, GLAT_arr, lonlat=True, nest=False)

# ===================== 步骤 4：逐像素加权 H0 拟合 =====================
print("\n" + "=" * 60)
print("步骤 4：逐像素加权线性回归（截距=0）")
print("=" * 60)

h0_map = np.full(npix, np.nan)
mask_map = np.zeros(npix, dtype=np.int32)

# 按像素分组
pix_groups = {}
for i in range(n_valid):
    pix = pix_indices[i]
    if pix not in pix_groups:
        pix_groups[pix] = []
    pix_groups[pix].append(i)

valid_pix_count = 0
for pix, idx_list in pix_groups.items():
    if len(idx_list) < MIN_GROUPS_PER_PIX:
        continue

    d_pix = d_arr[idx_list]
    v_pix = Vcmb_arr[idx_list]
    sig_pix = sigma_d_arr[idx_list]

    # 权重 w = 1 / sigma_d^2
    w_pix = 1.0 / (sig_pix ** 2)

    # H0 = sum(w_i * d_i * V_i) / sum(w_i * d_i^2)
    numerator = np.sum(w_pix * d_pix * v_pix)
    denominator = np.sum(w_pix * d_pix ** 2)

    if denominator <= 0:
        continue

    h0_val = numerator / denominator

    # 异常值剔除
    if h0_val < H0_MIN or h0_val > H0_MAX:
        continue

    h0_map[pix] = h0_val
    mask_map[pix] = 1
    valid_pix_count += 1

print(f"有效像素数（>={MIN_GROUPS_PER_PIX} 个星系群）: {valid_pix_count}")

# H0 统计
h0_valid = h0_map[~np.isnan(h0_map)]
print(f"H0 中位值: {np.median(h0_valid):.3f} km/s/Mpc")
print(f"H0 标准差: {np.std(h0_valid):.3f} km/s/Mpc")
print(f"H0 最小值: {np.min(h0_valid):.3f} km/s/Mpc")
print(f"H0 最大值: {np.max(h0_valid):.3f} km/s/Mpc")

# ===================== 步骤 5：群组面密度图 =====================
print("\n" + "=" * 60)
print("步骤 5：生成群组面密度图")
print("=" * 60)

density_map = np.zeros(npix)
for pix in pix_indices:
    density_map[pix] += 1

print(f"密度最大值: {int(density_map.max())} 群/像素")
print(f"密度中位数: {np.median(density_map[density_map > 0]):.1f} 群/像素")

# ===================== 步骤 6：相关性计算 =====================
print("\n" + "=" * 60)
print("步骤 6：H0 vs 密度 相关性分析")
print("=" * 60)

valid_mask = mask_map == 1
h0_vals = h0_map[valid_mask]
density_vals = density_map[valid_mask]
log_density = np.log10(density_vals + 1.0)

# Spearman 秩相关
spearman_r, spearman_p = stats.spearmanr(log_density, h0_vals)
# Pearson 相关
pearson_r, pearson_p = stats.pearsonr(log_density, h0_vals)

print(f"Spearman r = {spearman_r:.4f}, p = {spearman_p:.4e}")
print(f"Pearson  r = {pearson_r:.4f}, p = {pearson_p:.4e}")

# ===================== 步骤 7：写入输出文件 =====================
print("\n" + "=" * 60)
print("步骤 7：写入 FITS 文件和散点图")
print("=" * 60)

# 写入 H0 地图
hp.write_map(H0_MAP_FILE, h0_map, nest=False, coord="G", overwrite=True)
print(f"已写入: {H0_MAP_FILE}")

# 写入密度地图
hp.write_map(DENSITY_MAP_FILE, density_map, nest=False, coord="G", overwrite=True)
print(f"已写入: {DENSITY_MAP_FILE}")

# 写入掩模
hp.write_map(MASK_MAP_FILE, mask_map, nest=False, coord="G", overwrite=True)
print(f"已写入: {MASK_MAP_FILE}")

# 散点图
fig, ax = plt.subplots(figsize=(8, 6))

# 使用 hexbin 展示密度
hb = ax.hexbin(log_density, h0_vals, gridsize=30, cmap="viridis", mincnt=1)

# 拟合趋势线
z = np.polyfit(log_density, h0_vals, 1)
p = np.poly1d(z)
x_line = np.linspace(log_density.min(), log_density.max(), 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f"Linear fit (slope={z[0]:.2f})")

ax.set_xlabel(r"$\log_{10}(\text{Group Count} + 1)$", fontsize=12)
ax.set_ylabel(r"$H_0$ [km/s/Mpc]", fontsize=12)
ax.set_title("Local $H_0$ vs. Group Surface Density (CF4 table3)", fontsize=13)

# 标注相关系数
textstr = (
    f"Spearman: r={spearman_r:.3f}, p={spearman_p:.2e}\n"
    f"Pearson:  r={pearson_r:.3f}, p={pearson_p:.2e}\n"
    f"Valid pixels: {valid_pix_count}"
)
props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=props)

ax.legend(loc="lower left")
cbar = plt.colorbar(hb, ax=ax)
cbar.set_label("Pixel count", rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig(SCATTER_PLOT_FILE, dpi=300, bbox_inches="tight")
print(f"已写入: {SCATTER_PLOT_FILE}")

# ===================== 最终摘要 =====================
print("\n" + "=" * 60)
print("处理完成摘要")
print("=" * 60)
print(f"有效星系群数量:     {n_valid}")
print(f"有效 H0 像素数:       {valid_pix_count}")
print(f"H0 中位值:           {np.median(h0_valid):.3f} km/s/Mpc")
print(f"H0 标准差:           {np.std(h0_valid):.3f} km/s/Mpc")
print(f"H0 范围:             [{np.min(h0_valid):.3f}, {np.max(h0_valid):.3f}] km/s/Mpc")
print(f"Spearman r = {spearman_r:.4f}, p = {spearman_p:.4e}")
print(f"Pearson  r = {pearson_r:.4f}, p = {pearson_p:.4e}")
print("=" * 60)

if spearman_r < 0 and spearman_p < 0.05:
    print("ZGD 预言方向一致：密度越高 -> H0 越低（负相关显著）")
elif spearman_r < 0:
    print("方向一致但统计显著性不足（p >= 0.05）")
else:
    print("方向与 ZGD 预言相反")