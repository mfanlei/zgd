#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESI DR2 BAO 数据加载与官方值交叉验证
参考论文: arXiv:2503.14738 Table 4

用法: 把本脚本和以下两个文件放在同一目录下运行
  - desi_gaussian_bao_ALL_GCcomb_mean.txt
  - desi_gaussian_bao_ALL_GCcomb_cov.txt
"""

import numpy as np
import os

# =============================================================================
# 常数 & 配置
# =============================================================================
rd = 147.09       # 声学视界 (Mpc)
c  = 299792.458   # 光速 (km/s)

# 文件路径 (同目录)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEAN_FILE = os.path.join(SCRIPT_DIR, 'desi_gaussian_bao_ALL_GCcomb_mean.txt')
COV_FILE  = os.path.join(SCRIPT_DIR, 'desi_gaussian_bao_ALL_GCcomb_cov.txt')

# 类型名称 <-> 编码 映射
OBS_NAME_TO_CODE = {'DV_over_rs': 0, 'DM_over_rs': 1, 'DH_over_rs': 2}
OBS_CODE_TO_NAME = {0: 'DV_over_rs', 1: 'DM_over_rs', 2: 'DH_over_rs'}

# 官方值 (arXiv:2503.14738 Table 4)
OFFICIAL = [
    (7.942,  0.075), (13.587, 0.169), (21.863, 0.427),
    (17.347, 0.180), (19.458, 0.332), (21.574, 0.153),
    (17.641, 0.193), (27.605, 0.320), (14.178, 0.217),
    (30.519, 0.758), (12.816, 0.513), (8.632,  0.101),
    (38.988, 0.531),
]

# =============================================================================
# [1] 数据加载
# =============================================================================
def load_mean(filepath):
    """加载 mean 文件，返回 z, value, type_code 数组。
    文件格式: # 注释行; 数据行: z  value  quantity_name
    """
    z_list, val_list, typ_list = [], [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            z_list.append(float(parts[0]))
            val_list.append(float(parts[1]))
            typ_list.append(OBS_NAME_TO_CODE[parts[2]])
    return np.array(z_list), np.array(val_list), np.array(typ_list)


def load_cov(filepath, ndim):
    """加载协方差矩阵。文件中所有数字以空格/换行分隔，reshape 为 ndim×ndim。"""
    with open(filepath) as f:
        tokens = f.read().split()
    return np.array([float(t) for t in tokens]).reshape(ndim, ndim)


print("=" * 70)
print("  [1] 数据加载")
print("=" * 70)

z_arr, value_arr, type_arr = load_mean(MEAN_FILE)
n = len(z_arr)
print(f"  >> mean 文件: {n} 个数据点")

C = load_cov(COV_FILE, n)
print(f"  >> cov 文件:  {C.shape[0]}x{C.shape[1]}")

# =============================================================================
# [2] 数据验证
# =============================================================================
print("\n" + "=" * 70)
print("  [2] 数据验证")
print("=" * 70)

# 对称性 & 正定性
is_sym = np.allclose(C, C.T, atol=1e-15)
eigvals = np.linalg.eigvalsh(C)
is_pd = np.all(eigvals > 0)
print(f"  对称: {is_sym}  |  正定: {is_pd}")
print(f"  特征值范围: [{eigvals.min():.6e}, {eigvals.max():.6e}]")

# 13 个数据点列表
print(f"\n  {'#':>3}  {'z':>6}  {'类型':<12}  {'观测值':>14}  {'1σ误差':>10}")
print("  " + "-" * 55)
for i in range(n):
    sigma = np.sqrt(C[i, i])
    print(f"  {i+1:>3}  {z_arr[i]:>6.3f}  {OBS_CODE_TO_NAME[type_arr[i]]:<12}  "
          f"{value_arr[i]:>14.10f}  {sigma:>10.6f}")

# 非零非对角线协方差项及相关系数
print(f"\n  非零非对角线协方差项:")
print(f"  {'(i,j)':>8}  {'type_i':<12}  {'type_j':<12}  {'Cov_ij':>14}  {'ρ_ij':>8}")
print("  " + "-" * 60)
cnt = 0
for i in range(n):
    for j in range(i + 1, n):
        if abs(C[i, j]) > 1e-30:
            rho = C[i, j] / (np.sqrt(C[i, i]) * np.sqrt(C[j, j]))
            cnt += 1
            print(f"  ({i+1:2d},{j+1:2d})  {OBS_CODE_TO_NAME[type_arr[i]]:<12}  "
                  f"{OBS_CODE_TO_NAME[type_arr[j]]:<12}  {C[i,j]:>14.6e}  {rho:>8.4f}")
print(f"  共 {cnt} 项 (块对角结构: 仅同红移 DM-DH 有相关性)")

# z=2.33 顺序确认
ok12 = (abs(z_arr[11] - 2.33) < 0.01 and type_arr[11] == 2)
ok13 = (abs(z_arr[12] - 2.33) < 0.01 and type_arr[12] == 1)
print(f"\n  z=2.33 顺序确认:")
print(f"    第12个点: z={z_arr[11]}, {OBS_CODE_TO_NAME[type_arr[11]]}  → {'✓ DH' if ok12 else '✗'}")
print(f"    第13个点: z={z_arr[12]}, {OBS_CODE_TO_NAME[type_arr[12]]}  → {'✓ DM' if ok13 else '✗'}")

# =============================================================================
# [3] 与官方值逐点对比
# =============================================================================
print("\n" + "=" * 70)
print("  [3] 与官方值对比 (arXiv:2503.14738 Table 4)")
print("=" * 70)

print(f"\n  {'#':>3}  {'z':>6}  {'类型':<12}  {'文件值':>14}  {'官方值':>10}  "
      f"{'|Δ|':>10}  {'文件σ':>8}  {'官方σ':>8}  {'结果':>6}")
print("  " + "-" * 95)

all_pass = True
for i in range(n):
    off_mean, off_sigma = OFFICIAL[i]
    diff = abs(value_arr[i] - off_mean)
    ok = diff < 0.01
    if not ok:
        all_pass = False
    print(f"  {i+1:>3}  {z_arr[i]:>6.3f}  {OBS_CODE_TO_NAME[type_arr[i]]:<12}  "
          f"{value_arr[i]:>14.10f}  {off_mean:>10.6f}  {diff:>10.6f}  "
          f"{np.sqrt(C[i,i]):>8.6f}  {off_sigma:>8.6f}  "
          f"{'PASS' if ok else 'FAIL':>6}")

# =============================================================================
# [4] 总结
# =============================================================================
print("\n" + "=" * 70)
print("  [4] 验证总结")
print("=" * 70)

checks = [
    ("数据点总数 = 13",        n == 13),
    ("协方差矩阵 13x13",       C.shape == (13, 13)),
    ("协方差矩阵对称",         is_sym),
    ("协方差矩阵正定",         is_pd),
    ("所有点 |Δ| < 0.01",      all_pass),
    ("z=2.33 第12点 = DH",    ok12),
    ("z=2.33 第13点 = DM",    ok13),
]
for desc, result in checks:
    print(f"  [{'PASS' if result else 'FAIL'}] {desc}")

print(f"\n  常数: rd = {rd} Mpc, c = {c} km/s")

if all(r for _, r in checks):
    print("\n  ==============================")
    print("   ALL CHECKS PASSED!")
    print("  ==============================")
else:
    print("\n  ⚠️  部分验证未通过，请检查数据！")

print("=" * 70)
