#!/usr/bin/env python3
"""
ZGD 三维非线性正演：高分辨率修正版
"""
import numpy as np
import matplotlib.pyplot as plt

# ==================== 常数 ====================
G = 6.67430e-11
a0 = 2.04e-10
M_sun = 1.989e30
kpc = 3.086e19

# ==================== 网格设定 ====================
N = 128                     # 提高到 128
L_box = 120 * kpc           # 盒子缩小到 120 kpc
dx = L_box / N              # 分辨率约 0.94 kpc

x = np.linspace(-L_box/2, L_box/2, N)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

# ==================== 质量分布 ====================
R_d = 3 * kpc
z_d = 0.5 * kpc
M_disk = 5e10 * M_sun
r_cyl = np.sqrt(X**2 + Y**2)
rho_disk = (M_disk / (2 * np.pi * R_d**2 * z_d)) * np.exp(-r_cyl/R_d) * np.exp(-np.abs(Z)/z_d)

sigma_b = 1 * kpc
M_bulge = 1e10 * M_sun
r_sph = np.sqrt(X**2 + Y**2 + Z**2)
rho_bulge = (M_bulge / ((2*np.pi)**1.5 * sigma_b**3)) * np.exp(-r_sph**2 / (2*sigma_b**2))

rho_gal = rho_disk + rho_bulge

R_halo = 50 * kpc
M_halo = 2e12 * M_sun  
rho_halo = np.where(r_sph < R_halo, M_halo / (4/3 * np.pi * R_halo**3), 0.0)

rho_tot = rho_gal + rho_halo

# ==================== K空间准备 ====================
k1d = 2 * np.pi * np.fft.fftfreq(N, d=dx)
KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2
K2[0,0,0] = 1.0 

# ==================== 迭代求解 ====================
Phi = np.zeros((N, N, N))
alpha = 0.15  # 稍大一点的松弛因子
steps = 500

print("开始高分辨率三维非线性场迭代...")
for step in range(steps):
    Phi_k = np.fft.fftn(Phi)
    gx = -np.real(np.fft.ifftn(1j * KX * Phi_k))
    gy = -np.real(np.fft.ifftn(1j * KY * Phi_k))
    gz = -np.real(np.fft.ifftn(1j * KZ * Phi_k))
    
    g_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    g_mag_safe = np.maximum(g_mag, 1e-15)
    
    S = -4 * np.pi * G * a0 * rho_tot
    S_eff = S / g_mag_safe
    
    S_eff_k = np.fft.fftn(S_eff)
    S_eff_k[0,0,0] = 0.0
    Phi_new = np.real(np.fft.ifftn(-S_eff_k / K2))
    
    Phi = (1 - alpha) * Phi + alpha * Phi_new
    
    if step % 100 == 0:
        print(f"Step {step:3d} | max|g| = {g_mag.max():.2e}")

# ==================== 结果提取 ====================
Phi_k = np.fft.fftn(Phi)
gx = -np.real(np.fft.ifftn(1j * KX * Phi_k))
gy = -np.real(np.fft.ifftn(1j * KY * Phi_k))
gz = -np.real(np.fft.ifftn(1j * KZ * Phi_k))
g_mag = np.sqrt(gx**2 + gy**2 + gz**2)

r_test = 15 * kpc
idx_test = int((r_test + L_box/2) / dx)
g_test_simulated = g_mag[idx_test, N//2, N//2]

# 解析计算包围质量
# 盘（近似）：M_disk * (1 - (1 + r/Rd)exp(-r/Rd))
M_enc_disk = M_disk * (1 - (1 + r_test/R_d)*np.exp(-r_test/R_d))
# 核球：M_bulge * [erf(r/(sqrt(2)sigma)) - sqrt(2/pi)(r/sigma)exp(-r^2/(2sigma^2))]
from scipy.special import erf
M_enc_bulge = M_bulge * (erf(r_test/(np.sqrt(2)*sigma_b)) - np.sqrt(2/np.pi)*(r_test/sigma_b)*np.exp(-r_test**2/(2*sigma_b**2)))
# 外部晕
if r_test < R_halo:
    M_enc_halo = M_halo * (r_test / R_halo)**3
else:
    M_enc_halo = M_halo

M_enc_analytic = M_enc_disk + M_enc_bulge + M_enc_halo
g_test_theory = np.sqrt(G * M_enc_analytic * a0) / r_test

print("\n" + "="*40)
print("高分辨率正演结果 (r = 15 kpc, 盘平面)")
print("="*40)
print(f"网格分辨率: {dx/kpc:.2f} kpc")
print(f"解析包围质量: {M_enc_analytic/M_sun:.2e} M_sun")
print(f"球对称理论值 (无环境稀释): {g_test_theory:.3e} m/s^2")
print(f"三维模拟实际感受值:       {g_test_simulated:.3e} m/s^2")
print(f"直接场强比值 g_sim / g_theory: {g_test_simulated/g_test_theory:.3f}")
print("="*40)

# 绘图
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
im = ax.imshow(np.log10(g_mag[:, :, N//2].T + 1e-20), origin='lower', extent=[-L_box/2/kpc, L_box/2/kpc, -L_box/2/kpc, L_box/2/kpc], cmap='viridis')
ax.set_title('log10 |g_geo| (Z=0 plane) - High Res')
ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
plt.colorbar(im, label='log10 g [m/s^2]')
plt.savefig('zgd_3d_slice_highres.png', dpi=150)
plt.show()