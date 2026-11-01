#!/usr/bin/env python3
"""
星系团核心：BCG/ICM 质量比分界值扫描 + 图表生成
==========================================
固定总重子质量，扫描 BCG 与 ICM 的质量分配，
寻找 alpha_geo 恰好填平缺口的临界点，并绘制图表。
"""
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G = 6.67430e-11
c = 2.99792458e8
M_sun = 1.98892e30
kpc = 3.085677581e19
RP = 4.4e26
A0_ZGD = c**2 / RP

# ============ 质量模型 ============
def build_mass_interpolator(M_BCG, a_BCG, M_ICM, r_c_ICM, beta, R_max=3000.0):
    """构建 M_bar(<r) 的快速插值器"""
    r_grid = np.logspace(np.log10(0.1), np.log10(R_max), 500)
    
    # Hernquist (BCG)
    x_bcg = r_grid / a_BCG
    M_bcg_arr = M_BCG * (x_bcg / (1 + x_bcg))**2

    # Beta (ICM)
    def integrand(x):
        return x**2 * (1 + x**2)**(-1.5*beta)
    
    X = r_grid / r_c_ICM
    X_max = R_max / r_c_ICM
    total = quad(integrand, 0, X_max, limit=200)[0]
    M_icm_arr = np.array([quad(integrand, 0, min(x, X_max), limit=100)[0] for x in X]) * M_ICM / total
    
    M_bar_arr = M_bcg_arr + M_icm_arr
    return PchipInterpolator(r_grid, M_bar_arr)

# ============ 偏折角计算 ============
def alpha_N(b_kpc, M_interp):
    M_b = M_interp(b_kpc) * M_sun
    return 4 * G * M_b / (c**2 * b_kpc * kpc)

def alpha_geo(b_kpc, M_interp, a_0):
    b_m = b_kpc * kpc
    def integrand(z_m):
        r_m = np.sqrt(b_m**2 + z_m**2)
        r_kpc = r_m / kpc
        M_r = M_interp(r_kpc) * M_sun
        return np.sqrt(G * M_r * a_0) / r_m**2
    integral, _ = quad(integrand, -10*b_m, 10*b_m, limit=200, epsrel=1e-8)
    return 2 * b_m / c**2 * integral

# ============ 扫描主程序 + 画图 ============
def run_scan_and_plot():
    M_tot_bar = 5e12   # 总重子质量固定 5e12 M_sun
    f_bar = 0.20
    r_c = 150.0
    beta = 0.6
    a_BCG = 20.0
    b_test = 100.0

    print(f"{'M_BCG/M_ICM':<15}{'alpha_N':<12}{'alpha_geo':<12}{'alpha_obs':<12}{'geo/gap':<12}{'残余%':<10}")
    print("-" * 73)

    # 准备列表收集数据用于画图
    ratios = np.logspace(np.log10(0.1), np.log10(10), 25) # 增加取样点让曲线更平滑
    ratios_list = []
    geo_gap_list = []
    residual_list = []

    for ratio in ratios:
        M_BCG = M_tot_bar * ratio / (1 + ratio)
        M_ICM = M_tot_bar / (1 + ratio)
        
        M_interp = build_mass_interpolator(M_BCG, a_BCG, M_ICM, r_c, beta)
        
        def M_obs_interp(r): return M_interp(r) / f_bar
        aN = alpha_N(b_test, M_interp) * 206265
        aG = alpha_geo(b_test, M_interp, A0_ZGD) * 206265
        aO = alpha_N(b_test, M_obs_interp) * 206265 # 观测总偏折
        
        gap = aO - aN
        ratio_fill = aG / gap if gap > 0 else np.nan
        residual = (aO - aN - aG) / aO * 100 if aO > 0 else np.nan
        
        print(f"{ratio:<15.2f}{aN:<12.3f}{aG:<12.3f}{aO:<12.3f}{ratio_fill:<12.3f}{residual:<10.1f}")
        
        # 收集数据
        ratios_list.append(ratio)
        geo_gap_list.append(ratio_fill)
        residual_list.append(residual)

    # ============ 绘图 ============
    ratios_arr = np.array(ratios_list)
    geo_gap_arr = np.array(geo_gap_list)
    residual_arr = np.array(residual_list)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 主图：geo/gap 曲线
    ax1.plot(ratios_arr, geo_gap_arr, 'o-', color='tab:blue', lw=2, label='Geometric term / Gap')
    ax1.axhline(1.0, color='k', ls='--', lw=1.5, label='Perfect fill')
    
    # 阴影：完美吻合的区间 (0.95 - 1.05)
    ax1.fill_between([0.1, 10], 0.95, 1.05, color='green', alpha=0.15, label='5% error band')
    
    # 辅助线：X轴标明临界点
    ax1.axvline(1.39, color='tab:orange', ls=':', lw=1.5)
    ax1.axvline(1.93, color='tab:orange', ls=':', lw=1.5)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('$M_{\\mathrm{BCG}} / M_{\\mathrm{ICM}}$ (Mass ratio)')
    ax1.set_ylabel('$\\alpha_{\\mathrm{geo}} / (\\alpha_{\\mathrm{obs}} - \\alpha_N)$', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0.5, 3.2)
    
    # 副图：残余缺口 (%)
    ax2 = ax1.twinx()
    ax2.plot(ratios_arr, residual_arr, 's--', color='tab:red', alpha=0.6, label='Residual gap (%)')
    ax2.set_ylabel('Residual gap (%)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.axhline(0, color='tab:red', ls=':', lw=1.0, alpha=0.5)

    # 图例合并
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    ax1.set_title('Cluster Core Lensing: Gap Filling vs. Mass Concentration')
    ax1.grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('lensing_mass_ratio_scan.png', dpi=200)
    plt.close()
    print("\n图已保存：lensing_mass_ratio_scan.png")

if __name__ == "__main__":
    run_scan_and_plot()