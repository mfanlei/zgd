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
    val, _ = quad(lambda zp: 1.0 / ((1+zp) * H_LCDM(zp)), z, 100)
    # [FIX ①] 去掉多余的 1/H0_bare；val 单位已是 Mpc/(km/s)
    return val * 978.0

t_age = np.vectorize(t_age_scalar)

def D_z_scalar(z):
    Om_z = Om / (Om + (1-Om)/(1+z)**3)
    OL_z = 1 - Om_z
    g = 2.5*Om_z / (Om_z**(4/7) - OL_z + (1+Om_z/2)*(1+OL_z/70))
    return g * (1+z)**(-1)

D_z = np.vectorize(D_z_scalar)

def dDdt_scalar(z):
    dz = 1e-4
    D1 = D_z_scalar(z + dz)
    D2 = D_z_scalar(z - dz)
    dDdz = (D1 - D2) / (2 * dz)
    # [FIX ④] 真值 dt/dz = -1/((1+z)H)
    dtdz = -1.0 / ((1 + z) * H_LCDM(z))
    return dDdz / dtdz

dDdt = np.vectorize(dDdt_scalar)

# ================= 积分模型 =================
def model_delta_scalar(z, A, tau):
    def integrand(zp):
        rate = dDdt_scalar(zp)
        # [FIX ②] 核方向：zp>z 时 Delta>0，exp(-Delta/tau) 衰减
        Delta = t_age_scalar(z) - t_age_scalar(zp)
        # [FIX ③] 补 1/(1+zp) 换元因子
        return rate * np.exp(-Delta / tau) / ((1+zp) * H_LCDM(zp))
    
    integral, _ = quad(integrand, z, 20.0, limit=100)
    # [FIX ④] dtdz 翻正后 dDdt 为正，负号不再需要
    return A * integral

model_delta = np.vectorize(model_delta_scalar)

# ================= 数据准备 =================
z_data = np.linspace(0.01, 2.33, 50)
delta_data = delta_DPL(z_data)

# ================= 拟合 =================
popt, pcov = curve_fit(model_delta, z_data, delta_data, p0=[0.1, 5.0], bounds=([0.001, 0.1], [10, 50]))
A_fit, tau_fit = popt

# ================= 外推至 z=3.0 =================
delta_3_pred = model_delta(3.0, A_fit, tau_fit)
H_3_pred = H_bare(3.0) * (1 - delta_3_pred)
DH_3_pred = c / H_3_pred
DH_rd_3_pred = DH_3_pred / rd

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

# ================= [新增] 端点输出，用于核对 20.7% / 5.3% 的出处 =================
# tau->0 应对应"纯瞬时"(短核)；tau->inf 应对应"纯累积"(长核)
for tau_test, label in [(0.01, "tau->0 (instantaneous)"), (1000.0, "tau->inf (cumulative)")]:
    integ_val_233 = quad(
        lambda zp: dDdt_scalar(zp) * np.exp(-(t_age_scalar(2.33)-t_age_scalar(zp))/tau_test) / ((1+zp)*H_LCDM(zp)),
        2.33, 20.0, limit=100
    )[0]
    print(f"{label}: raw integral at z=2.33 = {integ_val_233:.6e}")
    print(f"    (未乘 A；若要和论文 20.7%/5.3% 对比，需乘同一 A 并与 DPL 锚点 -0.1% 比)")
print("="*50)

# 画图
z_plot = np.linspace(0.01, 3.0, 100)
plt.plot(z_plot, delta_DPL(z_plot), 'b--', label='DPL Phenomenology')
plt.plot(z_plot, model_delta(z_plot, A_fit, tau_fit), 'r-', label=f'Integral Model (A={A_fit:.2f}, tau={tau_fit:.1f})')
plt.axvline(2.33, color='k', linestyle=':', label='DR2 Limit')
plt.axhline(0, color='gray', linestyle='-')
plt.xlabel('Redshift z')
plt.ylabel(r'Renormalization Factor \(\delta(z)\)')
plt.legend()
plt.title('Time-Integrated Renormalization')
plt.savefig('delta_time_integrated.png', dpi=200)

# ================= [诊断] 固定 tau，单参数拟合 A，输出锚点预测 =================
from scipy.optimize import curve_fit as cf

z_diag = np.linspace(0.01, 2.33, 30)
delta_diag = delta_DPL(z_diag)

for tau_fixed, label in [(0.01, "short kernel tau=0.01"),
                          (4.35, "intermediate tau=4.35"),
                          (1000.0, "long kernel tau=1000")]:
    model_fixed = lambda z, A: model_delta(z, A, tau_fixed)
    try:
        popt_A, _ = cf(model_fixed, z_diag, delta_diag,
                       p0=[0.1], bounds=([0.0001], [100.0]))
        A_only = popt_A[0]
        d233 = model_delta(2.33, A_only, tau_fixed)
        d3 = model_delta(3.0, A_only, tau_fixed)
        print(f"{label}: A={A_only:.5f}, delta(2.33)={d233*100:+.2f}%, delta(3.0)={d3*100:+.2f}%")
    except Exception as e:
        print(f"{label}: fit failed -> {e}")

# ================= 集中核验证（核对 §5.9 加固句，修复版） =================

def K_gauss(Delta, Delta0, sigma_k):
    return np.exp(-(Delta - Delta0)**2 / (2 * sigma_k**2))

def model_delta_concentrated_scalar(z, A, Delta0, sigma_k):
    def integrand(zp):
        rate = dDdt_scalar(zp)
        Delta = t_age_scalar(z) - t_age_scalar(zp)
        return rate * K_gauss(Delta, Delta0, sigma_k) / ((1+zp) * H_LCDM(zp))
    integral, _ = quad(integrand, z, 20.0, limit=200)
    return A * integral

Delta0 = 4.8
z_fit = np.linspace(0.01, 2.33, 30)
delta_fit = delta_DPL(z_fit)

print("\n固定 Δ_0 = 4.8 Gyr，单参数拟合 A：")
print(f"{'σ_k(Gyr)':<10} {'A_fit':<12} {'δ(0.01)':<12} {'δ(0.93)':<12} {'δ(1.4)':<12} {'δ(2.33)':<12}")
for sigma_k in [0.3, 0.5, 1.0, 2.0]:
    def model_vec(z, A):
        return model_delta_concentrated_scalar(z, A, Delta0, sigma_k)
    model_vec_np = np.vectorize(model_vec)
    try:
        popt_A, _ = curve_fit(model_vec_np, z_fit, delta_fit, p0=[1.0], bounds=([-100], [100]))
        A = popt_A[0]
    except Exception as e:
        print(f"  σ_k={sigma_k}: 拟合失败: {e}")
        continue
    d001 = model_delta_concentrated_scalar(0.01, A, Delta0, sigma_k)
    d093 = model_delta_concentrated_scalar(0.93, A, Delta0, sigma_k)
    d14  = model_delta_concentrated_scalar(1.4,  A, Delta0, sigma_k)
    d233 = model_delta_concentrated_scalar(2.33, A, Delta0, sigma_k)
    print(f"{sigma_k:<10.1f} {A:<12.4f} {d001*100:<+12.2f} {d093*100:<+12.2f} {d14*100:<+12.2f} {d233*100:<+12.2f}")

print("\nDPL 廓线参考值：")
for z in [0.01, 0.93, 1.4, 2.33]:
    print(f"  z={z:.2f}: δ_DPL={delta_DPL(z)*100:+.2f}%")
