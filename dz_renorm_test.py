import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ================= 常数 =================
c = 299792.458  # km/s
rd = 147.09     # Mpc
H0_bare = 70.85 # km/s/Mpc

# DPL 最佳拟合参数（来自论文）
H0_fit = 64.84
alpha = 1000.0
beta = 22.26
gamma = 13.23
delta_param = 0.0650

def H_DPL(z):
    return H0_fit * (((1+z)**beta + alpha*(1+z)**gamma) / (1+alpha))**delta_param

def H_bare(z):
    return H0_bare * (1+z)

def delta_DPL(z):
    """定义重整化因子 delta(z) = 1 - H_DPL / H_bare"""
    return 1 - H_DPL(z) / H_bare(z)

# ================= 结构增长因子 (代表宇宙网演化) =================
Om = 0.315
def D_z(z):
    """Carroll, Press & Turner 近似的线性结构增长因子"""
    g_z = Om / (Om + (1-Om)/(1+z)**3)
    return (g_z ** (-0.23)) * (1+z)**(-1) * (1/1.0) # z=0 归一化为 1

# ================= 数据准备 (DR2 窗口) =================
z_data = np.linspace(0.01, 2.33, 100)
delta_data = delta_DPL(z_data)
D_data = np.array([D_z(z) for z in z_data])

# ================= 物理模型拟合 =================
# 假设 delta(z) 是结构增长因子的非线性泛函
def model_delta(D, A, B, C):
    return A * D + B * D**2 + C * D**3

popt, _ = curve_fit(model_delta, D_data, delta_data)
A_fit, B_fit, C_fit = popt

# ================= 外推至 z=3.0 =================
D_3 = D_z(3.0)
delta_3_pred = model_delta(D_3, A_fit, B_fit, C_fit)
H_3_pred = H_bare(3.0) * (1 - delta_3_pred)

# 计算 D_H / rd
DH_3_pred = c / H_3_pred
DH_rd_3_pred = DH_3_pred / rd

# DPL 自身的严格外推值
H_3_dpl = H_DPL(3.0)
DH_rd_3_dpl = (c / H_3_dpl) / rd

print("="*50)
print("基于结构增长因子的非线性场演化外推")
print("="*50)
print(f"拟合函数：delta(D) = {A_fit:.4f}D + {B_fit:.4f}D^2 + {C_fit:.4f}D^3")
print(f"D(z=3.0) = {D_3:.4f}")
print(f"外推 delta(z=3.0) = {delta_3_pred:+.4f}")
print(f"外推 H(z=3.0)     = {H_3_pred:.2f} km/s/Mpc")
print("-"*50)
print(f"纯物理外推 D_H/rd(3.0) = {DH_rd_3_pred:.3f}")
print(f"DPL 模型 D_H/rd(3.0)   = {DH_rd_3_dpl:.3f}")
print("="*50)

# 画图对比
z_plot = np.linspace(0.01, 3.0, 100)
plt.plot(z_plot, [model_delta(D_z(z), *popt) for z in z_plot], 'r-', label='Structure-driven EFT Model')
plt.plot(z_plot, [delta_DPL(z) for z in z_plot], 'b--', label='DPL Phenomenology')
plt.axvline(2.33, color='k', linestyle=':', label='DR2 Limit')
plt.axhline(0, color='gray', linestyle='-')
plt.xlabel('Redshift z')
plt.ylabel('Renormalization Factor \(\delta(z)\)')
plt.legend()
plt.title('Cosmic-web driven renormalization')
plt.savefig('delta_structure_driven.png', dpi=200)
