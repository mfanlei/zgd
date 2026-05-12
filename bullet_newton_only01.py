#!/usr/bin/env python3
"""
纯牛顿对照组：子弹星系团偏移量计算
-----------------------------------
禁用 ZGD 几何修正项 (g_geo = 0)，仅使用牛顿引力 g_N，
计算有效质量密度 ρ_eff 的峰值位置，作为 ZGD 结果的对照基准。

运行方式： python bullet_newton_only.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter

# ==================== 物理常数 ====================
G  = 6.67430e-11          # m^3 kg^-1 s^-2
MSUN = 1.98847e30         # kg
MPC  = 3.085677581e22     # m
KPC  = MPC / 1000.0       # m

# ==================== β 模型（与原版相同） ====================
def beta_model_rho0(M_total, rc, beta):
    """计算 β 模型的中心密度 ρ₀，使总质量等于 M_total"""
    x = np.logspace(-6, 4, 100000)
    integrand = x**2 / (1 + x**2)**(1.5*beta)
    integral = np.trapezoid(integrand, x)
    rho0 = M_total / (4 * np.pi * rc**3 * integral)
    return rho0

_RHO0_CACHE = {}
def get_rho0(M_total, rc, beta):
    key = (M_total, rc, beta)
    if key not in _RHO0_CACHE:
        _RHO0_CACHE[key] = beta_model_rho0(M_total, rc, beta)
    return _RHO0_CACHE[key]

def M_beta_fast(r, M_total, rc, beta):
    """包围质量快速近似（与原版相同）"""
    x = r / rc
    if beta == 0.7:
        a, b, c = 2.15, 0.82, 1.05
    elif beta == 0.8:
        a, b, c = 2.45, 0.75, 1.08
    else:
        a, b, c = 3.0*beta - 0.6, 0.8, 1.0
    M = M_total * (1.0 - 1.0 / (1.0 + x**a)**b)**c
    return np.clip(M, 0.0, M_total)

# ==================== 纯牛顿引力场（关键改动） ====================
def newton_gravity_field(X, Y, M, xc, yc, rc, beta):
    """
    仅计算牛顿引力 (g_geo = 0)
    返回 gx, gy, g_mag, M_enc
    """
    dx = X - xc
    dy = Y - yc
    r = np.sqrt(dx**2 + dy**2)
    r_safe = np.clip(r, 1e3, None)

    M_enc = M_beta_fast(r_safe, M, rc, beta)
    g_mag = G * M_enc / r_safe**2          # 纯牛顿 g_N

    # 矢量分量（指向质量中心）
    gx = -g_mag * dx / r_safe
    gy = -g_mag * dy / r_safe

    return gx, gy, g_mag, M_enc

def effective_mass_density(gx, gy, dx):
    div_g = np.gradient(gx, dx, axis=1) + np.gradient(gy, dx, axis=0)
    return -div_g / (4.0 * np.pi * G)

# ==================== 子弹星系团模型（仅牛顿） ====================
class BulletClusterNewton:
    OBSERVED_OFFSET_MPC = 0.6
    SCALE_KPC_PER_ARCSEC = 4.8

    def __init__(self, M1=1.5e14, M2=5e13, rc1=150, rc2=50, 
                 beta1=0.7, beta2=0.8, bullet_offset_arcsec=100,
                 grid_size_mpc=3.0, grid_n=400):
        self.M1 = M1 * MSUN
        self.M2 = M2 * MSUN
        self.rc1 = rc1 * KPC
        self.rc2 = rc2 * KPC
        self.beta1 = beta1
        self.beta2 = beta2
        self.bullet_x = bullet_offset_arcsec * self.SCALE_KPC_PER_ARCSEC * KPC
        self.bullet_y = 0.0

        self.L = grid_size_mpc * MPC
        self.N = grid_n
        self.dx = self.L / self.N

        x = np.linspace(-self.L/2, self.L/2, self.N)
        y = np.linspace(-self.L/2, self.L/2, self.N)
        self.X, self.Y = np.meshgrid(x, y)

        self._compute_fields()

    def _compute_fields(self):
        # 主团牛顿场
        self.gx1, self.gy1, self.g1_mag, self.M1_enc = newton_gravity_field(
            self.X, self.Y, self.M1, 0, 0, self.rc1, self.beta1)
        # 次团牛顿场
        self.gx2, self.gy2, self.g2_mag, self.M2_enc = newton_gravity_field(
            self.X, self.Y, self.M2, self.bullet_x, self.bullet_y, 
            self.rc2, self.beta2)

        self.gx_total = self.gx1 + self.gx2
        self.gy_total = self.gy1 + self.gy2
        self.g_mag = np.sqrt(self.gx_total**2 + self.gy_total**2)

        self.rho_eff = effective_mass_density(self.gx_total, self.gy_total, self.dx)
        self.rho_eff_smooth = gaussian_filter(self.rho_eff, sigma=3)

    def find_peaks(self):
        max_idx = np.unravel_index(np.argmax(self.rho_eff_smooth), 
                                   self.rho_eff_smooth.shape)
        self.peak_x = self.X[max_idx]
        self.peak_y = self.Y[max_idx]
        self.peak_rho = self.rho_eff_smooth[max_idx]

        self.offset_m = np.sqrt(self.peak_x**2 + self.peak_y**2)
        self.offset_mpc = self.offset_m / MPC
        self.offset_arcsec = self.offset_m / (self.SCALE_KPC_PER_ARCSEC * KPC)

        # 局部极大值
        local_max = (self.rho_eff_smooth == 
                     maximum_filter(self.rho_eff_smooth, size=20))
        coords = np.argwhere(local_max)
        self.local_peaks = []
        for cx, cy in coords:
            px, py = self.X[cx,cy]/MPC, self.Y[cx,cy]/MPC
            dist = np.sqrt(px**2 + py**2)
            self.local_peaks.append({
                'x_mpc': px, 'y_mpc': py,
                'dist_mpc': dist,
                'rho': self.rho_eff_smooth[cx,cy]
            })
        self.local_peaks.sort(key=lambda p: p['rho'], reverse=True)
        return self.offset_mpc

    def print_report(self):
        print("=" * 70)
        print("纯牛顿对照组：子弹星系团偏移量")
        print("=" * 70)
        print(f"\n[模型参数]")
        print(f"  主团质量 M1: {self.M1/MSUN:.2e} M☉")
        print(f"  次团质量 M2: {self.M2/MSUN:.2e} M☉")
        print(f"  主团核半径 rc1: {self.rc1/KPC:.0f} kpc")
        print(f"  次团核半径 rc2: {self.rc2/KPC:.0f} kpc")
        print(f"  次团预设位置: {self.bullet_x/MPC:.3f} Mpc")
        print(f"\n[峰值分析]")
        print(f"  全局峰值: ({self.peak_x/MPC:.4f}, {self.peak_y/MPC:.4f}) Mpc")
        print(f"  峰值密度: {self.peak_rho:.3e} kg/m³")
        print(f"  偏移量: {self.offset_mpc:.4f} Mpc = {self.offset_arcsec:.1f} 角秒")
        print(f"  观测偏移 (Clowe+06): {self.OBSERVED_OFFSET_MPC:.1f} Mpc")
        print(f"\n[局部极大值 TOP 3]")
        for i, p in enumerate(self.local_peaks[:3], 1):
            print(f"  {i}. ({p['x_mpc']:.3f}, {p['y_mpc']:.3f}) Mpc, "
                  f"距原点{p['dist_mpc']:.3f} Mpc, ρ={p['rho']:.3e}")
        print("=" * 70)


if __name__ == "__main__":
    model = BulletClusterNewton(grid_n=400)
    model.find_peaks()
    model.print_report()