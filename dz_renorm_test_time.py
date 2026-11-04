import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad
import matplotlib.pyplot as plt

# ================= 常数 =================
c = 299792.458  # km/s
rd = 147.09     # Mpc
H0_bare = 70.85 # km/s/Mpc
Om = 0.315
OL = 1 - Om

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
    return 1 - H_DPL(z) / H_bare(z)

def H_LCDM(z):
    return H0_bare * np.sqrt(Om * (1+z)**3 + OL)

# 将 z 转为 t(z) 的标量函数
def t_age_scalar(z):
    # 积分从 z 到 100，避免无穷大
    val, _ = quad(lambda zp: 1.0 / ((1+zp) * H_LCDM(zp)), z, 100)
    # 1/H0 的单位是 13.79 Gyr，Mpc/(km/s) 转换到 Gyr 大约是 978
    return val * (1.0 / H0_bare) * 978.0

# 包装为向量化函数
t_age = np.vectorize(t_age_scalar)

def D_z_scalar(z):
    g_z = Om / (Om + (1-Om)/(1+z)**3)
    return (g_z ** (-0.23)) * (1+z)**(-1) * (1/1.0)

D_z = np.vectorize(D_z_scalar)

def dDdt_scalar(z):
    # 计算 D(z) 对 t 的导数，即结构形成率
    dz = 1e-4
    D1 = D_z_scalar(z + dz)
    D2 = D_z_scalar(z - dz)
    dDdz = (D1 - D2) / (2 * dz)
    dtdz = 1.0 / ((1 + z) * H_LCDM(z))
    return dDdz / dtdz

dDdt = np.vectorize(dDdt_scalar)

# ================= 积分模型 =================
def model_delta_scalar(z, A, tau):
    def integrand(zp):
        rate = dDdt_scalar(zp) # 用结构形成率驱动
        dt = t_age_scalar(zp) - t_age_scalar(z)
        # 如果时间太久远（dt很大），被exp抑制
        return rate * np.exp(-dt / tau) / H_LCDM(zp)
    
    integral, _ = quad(integrand, z, 20.0, limit=100)
    # A 为负，因为形成过程产生阻力，delta>0
    return -A * integral

model_delta = np.vectorize(model_delta_scalar)

# ================= 数据准备 =================
z_data = np.linspace(0.01, 2.33, 50)
delta_data = delta_DPL(z_data)

# ================= 拟合 =================
# A 限制为正数即可（内部加了负号），tau 为时间尺度（Gyr）
popt, pcov = curve_fit(model_delta, z_data, delta_data, p0=[0.1, 5.0], bounds=([0.001, 0.1], [10, 50]))
A_fit, tau_fit = popt

# ================= 外推至 z=3.0 =================
delta_3_pred = model_delta(3.0, A_fit, tau_fit)
H_3_pred = H_bare(3.0) * (1 - delta_3_pred)
DH_3_pred = c / H_3_pred
DH_rd_3_pred = DH_3_pred / rd

# DPL 自身的严格外推值
H_3_dpl = H_DPL(3.0)
DH_rd_3_dpl = (c / H_3_dpl) / rd

print("="*50)
print("带时间积分的重整化模型 (记忆核拟合)")
print("="*50)
print(f"拟合参数: A = {A_fit:.4f}, tau = {tau_fit:.2f} Gyr")
print(f"锚点测试 delta(2.33) = {model_delta(2.33, A_fit, tau_fit)*100:.2f}% (目标: -0.1%)")
print(f"外推 H(z=3.0)     = {H_3_pred:.2f} km/s/Mpc")
print("-"*50)
print(f"纯物理外推 D_H/rd(3.0) = {DH_rd_3_pred:.3f}")
print(f"DPL 模型 D_H/rd(3.0)   = {DH_rd_3_dpl:.3f}")
print("="*50)

# 画图
z_plot = np.linspace(0.01, 3.0, 100)
plt.plot(z_plot, delta_DPL(z_plot), 'b--', label='DPL Phenomenology')
plt.plot(z_plot, model_delta(z_plot, A_fit, tau_fit), 'r-', label=f'Integral Model (A={A_fit:.2f}, tau={tau_fit:.1f})')
plt.axvline(2.33, color='k', linestyle=':', label='DR2 Limit')
plt.axhline(0, color='gray', linestyle='-')
plt.xlabel('Redshift z')
plt.ylabel('Renormalization Factor \(\delta(z)\)')
plt.legend()
plt.title('Time-Integrated Renormalization')
plt.savefig('delta_time_integrated.png', dpi=200)