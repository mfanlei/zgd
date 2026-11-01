#!/usr/bin/env python3
"""
星系团核心：不同重子分数 (f_bar) 下的 BCG/ICM 质量比扫描
==========================================
生成 2D 扫描表（供论文引用），并绘制三线对比图。
"""
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置字体和数学公式风格，使其更接近 LaTeX 渲染
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

G = 6.67430e-11
c = 2.99792458e8
M_sun = 1.98892e30
kpc = 3.085677581e19
RP = 4.4e26
A0_ZGD = c**2 / RP  # 保持 C=1 的约定

def build_mass_interpolator(M_BCG, a_BCG, M_ICM, r_c_ICM, beta, R_max=3000.0):
    r_grid = np.logspace(np.log10(0.1), np.log10(R_max), 500)
    x_bcg = r_grid / a_BCG
    M_bcg_arr = M_BCG * (x_bcg / (1 + x_bcg))**2

    def integrand(x):
        return x**2 * (1 + x**2)**(-1.5*beta)
    X = r_grid / r_c_ICM
    X_max = R_max / r_c_ICM
    total = quad(integrand, 0, X_max, limit=200)[0]
    M_icm_arr = np.array([quad(integrand, 0, min(x, X_max), limit=100)[0] for x in X]) * M_ICM / total
    
    M_bar_arr = M_bcg_arr + M_icm_arr
    return PchipInterpolator(r_grid, M_bar_arr)

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

def run_scan_and_plot():
    M_tot_bar = 5e12
    r_c = 150.0
    beta = 0.6
    a_BCG = 20.0
    b_test = 100.0

    f_bar_values = [0.20, 0.10, 0.05]
    colors = ['tab:blue', 'tab:orange', 'tab:red']
    # 修正：使用 raw string 和 $ 定界符
    labels = [r'Group core ($f_{\rm bar}=0.20$)', 
              r'Rich cluster ($f_{\rm bar}=0.10$)', 
              r'ICM-dominated ($f_{\rm bar}=0.05$)']
    
    ratios = np.logspace(np.log10(0.01), np.log10(10), 40)
    
    # 专门用于打印 2D 表的特征比例点
    table_ratios = [0.03, 0.05, 0.10, 1.00, 3.00] 
    table_data = {r: {} for r in table_ratios}

    fig, ax1 = plt.subplots(figsize=(9, 6))

    for f_bar, color, label in zip(f_bar_values, colors, labels):
        ratios_list = []
        geo_gap_list = []
        
        for ratio in ratios:
            M_BCG = M_tot_bar * ratio / (1 + ratio)
            M_ICM = M_tot_bar / (1 + ratio)
            M_interp = build_mass_interpolator(M_BCG, a_BCG, M_ICM, r_c, beta)
            
            def M_obs_interp(r): return M_interp(r) / f_bar
            
            aN = alpha_N(b_test, M_interp) * 206265
            aG = alpha_geo(b_test, M_interp, A0_ZGD) * 206265
            aO = alpha_N(b_test, M_obs_interp) * 206265 
            
            gap = aO - aN
            ratio_fill = aG / gap if gap > 0 else np.nan
            
            ratios_list.append(ratio)
            geo_gap_list.append(ratio_fill)
            
            # 记录插值到表格数据
            for tr in table_ratios:
                if abs(np.log10(ratio) - np.log10(tr)) < 0.05: # 粗略匹配
                    table_data[tr][f_bar] = ratio_fill

        ax1.plot(ratios_list, geo_gap_list, '-', color=color, lw=2, label=label)

    # 打印 2D 表格
    print("\n" + "="*75)
    print("2D 扫描表：几何项填补率 (\\(\\alpha_{\\rm geo} / \\Delta\\alpha\\))")
    print("="*75)
    print(f"{'M_BCG/M_ICM':<15} | {'f_bar=0.20 (Group)':<20} | {'f_bar=0.10 (Rich)':<20} | {'f_bar=0.05 (ICM)':<20}")
    print("-" * 75)
    for tr in table_ratios:
        row = f"{tr:<15.2f} | "
        for fb in [0.20, 0.10, 0.05]:
            # 如果没精确匹配上，用最近邻输出（或重新算一次精确点）
            val = table_data[tr].get(fb, np.nan)
            if np.isnan(val):
                # 如果没有对应点，直接在精确 tr 重算一下
                M_BCG_t = M_tot_bar * tr / (1 + tr)
                M_ICM_t = M_tot_bar / (1 + tr)
                Mi = build_mass_interpolator(M_BCG_t, a_BCG, M_ICM_t, r_c, beta)
                def Mo_t(r): return Mi(r) / fb
                aN_t = alpha_N(b_test, Mi) * 206265
                aG_t = alpha_geo(b_test, Mi, A0_ZGD) * 206265
                aO_t = alpha_N(b_test, Mo_t) * 206265 
                gap_t = aO_t - aN_t
                val = aG_t / gap_t if gap_t > 0 else np.nan
            row += f"{val:<20.3f} | "
        print(row)
    print("="*75 + "\n")

    ax1.axhline(1.0, color='k', ls='--', lw=1.5, label='Perfect fill (ratio = 1)')
    ax1.fill_between([0.01, 10], 0.9, 1.1, color='green', alpha=0.15, label='10% error band')
    # 修正：使用 $ 定界符和 raw string
    ax1.axvline(0.03, color='gray', ls=':', lw=2, 
                label=r'Typical rich cluster ($M_{\rm BCG}/M_{\rm ICM} \sim 0.03$)')

    ax1.set_xscale('log')
    # 修正：使用 $ 定界符和 raw string
    ax1.set_xlabel(r'$M_{\rm BCG} / M_{\rm ICM}$ (Mass ratio)')
    ax1.set_ylabel(r'Geometric term / Gap  ($\alpha_{\rm geo} / \Delta\alpha$)')
    ax1.set_title('Lensing Gap Filling vs. Baryon Fraction')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(alpha=0.3, which='both')
    ax1.set_ylim(0, 3.0)

    plt.tight_layout()
    plt.savefig('lensing_fbar_scan.png', dpi=200)
    plt.close()
    print("图已保存：lensing_fbar_scan.png")

if __name__ == "__main__":
    run_scan_and_plot()