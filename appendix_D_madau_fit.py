import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.linalg import solve

# ================= 1. 宇宙学背景 =================
c = 299792.458
Rp = 14260.0  # Mpc
H0_bare = 70.85
rd = 147.09

def H_bare(z): return H0_bare * (1+z)
def H_LCDM(z, Om): return H0_bare * np.sqrt(Om*(1+z)**3 + (1-Om))

def D_c(z, Om):
    if z <= 0: return 0.0
    val, _ = quad(lambda zp: c / H_LCDM(zp, Om), 0, z)
    return val

# ================= 2. Madau 结构组装函数 =================
def Madau(x, zt, a, b):
    return x**a / (1.0 + (x/zt)**(a+b))

# 计算未归一化的累积项
def raw_Sigma(z, Om, n, z_cut):
    if z <= 0: return 0.0
    def integrand(zp):
        Dc_zp = D_c(zp, Om)
        cutoff = max(0.0, 1.0 - (Dc_zp / Rp)**2)**n
        weight = (1 + zp)**1.5  # 物理的 sqrt(M) 标度
        f_struct = Madau(1+zp, z_cut, 2.0, 4.0)  # 结构稀少截断
        return weight * cutoff * f_struct * (c / H_LCDM(zp, Om))
    val, _ = quad(integrand, 0, z, limit=100)
    return val

# 计算 z=2.81 处的参考值，用于归一化
ref_ratio = raw_Sigma(2.81, 0.315, 3.0, 5.0) / np.log(1+2.81)

def delta_model(z, params):
    A_dil, z1, A_enh, n, Om, z_cut = params
    if z <= 0: return A_dil * Madau(1.0, z1, 0.7, 5.0)
    
    x = 1.0 + z
    # 1. 稀释项
    dil = A_dil * Madau(x, z1, 0.7, 5.0)
    
    # 2. 增强项（归一化后）
    ratio = raw_Sigma(z, Om, n, z_cut) / np.log(x)
    enh = A_enh * (ratio / ref_ratio)
    
    return dil - enh

def H_eff(z, params):
    d = delta_model(z, params)
    return H_bare(z) * (1 - d)

# ================= 3. 读取数据 =================
mean_file = 'desi_gaussian_bao_ALL_GCcomb_mean.txt'
cov_file = 'desi_gaussian_bao_ALL_GCcomb_cov.txt'
z_bao, obs_bao, types_bao = [], [], []
with open(mean_file, 'r') as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        parts = line.split()
        z_bao.append(float(parts[0]))
        obs_bao.append(float(parts[1]))
        types_bao.append(parts[2])
cov_bao = np.loadtxt(cov_file)

z_lya = np.array([2.13, 2.40, 2.81])
DH_lya_obs = np.array([9.40, 8.28, 7.22])
DH_err_lya = np.array([0.20, 0.17, 0.17])

delta_obs_lya = []
for i, z in enumerate(z_lya):
    H_obs = c / (DH_lya_obs[i] * rd)
    delta_obs_lya.append(1 - H_obs / H_bare(z))
delta_obs_lya = np.array(delta_obs_lya)

# ================= 4. 卡方 =================
def calc_chi2(params):
    # 参数边界：A_dil(0~0.3), z1(1~4), A_enh(0~0.2), n(1~4), Om(0.2~0.4), z_cut(3~8)
    if params[0] < 0 or params[0] > 0.3 or params[2] < 0 or params[2] > 0.2 or params[3] < 1 or params[3] > 4 or params[4] < 0.2 or params[4] > 0.4 or params[5] < 3 or params[5] > 8:
        return 1e6
    
    model_pred = []
    for z, t in zip(z_bao, types_bao):
        DM = D_c(z, params[4])
        DH = c / H_eff(z, params)
        DV = (z * DM**2 * DH)**(1/3)
        if t == 'DV_over_rs': model_pred.append(DV / rd)
        elif t == 'DM_over_rs': model_pred.append(DM / rd)
        elif t == 'DH_over_rs': model_pred.append(DH / rd)
        else: model_pred.append(0)
        
    chi2_bao = (np.array(model_pred) - np.array(obs_bao)) @ solve(cov_bao, (np.array(model_pred) - np.array(obs_bao)))
    
    DH_lya_pred = np.array([c / H_eff(z, params) / rd for z in z_lya])
    chi2_lya = np.sum(((DH_lya_pred - DH_lya_obs) / DH_err_lya)**2)
    
    return chi2_bao + chi2_lya

# ================= 5. 优化 =================
# 初值: [A_dil, z1, A_enh, n, Om, z_cut]
p0 = [0.10, 2.5, 0.05, 3.0, 0.315, 5.0]
res = minimize(calc_chi2, p0, method='Nelder-Mead', options={'maxiter': 5000})

print("\n拟合成功！")
print(f"最小卡方 Chi2 = {res.fun:.2f}")
print(f"参数: A_dil={res.x[0]:.4f}, z1={res.x[1]:.3f}, A_enh={res.x[2]:.4f}, n={res.x[3]:.3f}, Om={res.x[4]:.3f}, z_cut={res.x[5]:.3f}")

# ================= 6. 画图 =================
z_plot = np.linspace(0.01, 8.0, 300)
delta_plot = [delta_model(z, res.x) * 100 for z in z_plot]

plt.figure(figsize=(10, 6))
plt.plot(z_plot, delta_plot, 'b-', lw=2.5, label=r'Madau + $\sqrt{M}$ (Normalized)')
plt.axhline(0, color='black', linestyle='--', label='Bare Geometry (g=1)')
plt.axvline(2.33, color='orange', linestyle=':', label='DR2 Limit')
plt.errorbar(z_lya, delta_obs_lya * 100, yerr=np.array([2.0, 2.1, 2.4]), fmt='ro', capsize=4, label=r'Actual Ly$\alpha$ $\delta$')

plt.xlabel('Redshift z', fontsize=14)
plt.ylabel(r'$\delta(z)$ [%]', fontsize=14)
plt.title('ZGD: Madau Assembly & Normalized Causal Cutoff', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-15, 15)
plt.savefig('zgd_madau_fit_v2.png', dpi=150)
print("图像已保存为 zgd_madau_fit_v2.png")