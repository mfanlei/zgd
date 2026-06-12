import numpy as np
from scipy.integrate import cumulative_trapezoid

# ------------------------- 物理常数 -------------------------
c = 2.99792458e8          # 光速 (m/s)
l_P = 1.616255e-35        # 普朗克长度 (m)
T_now = 13.8e9 * 365.25 * 24 * 3600  # 宇宙年龄 (秒)，约 4.355e17 s
H0_bare = 1.0 / T_now     # 裸几何哈勃常数 (s^-1)，约为 2.296e-18 s^-1
H0_bare_kms = H0_bare * 3.085677581e19  # 转换为 km/s/Mpc，约 70.85

print(f"裸几何 H0 = {H0_bare_kms:.2f} km/s/Mpc")

# ------------------------- 红移网格 -------------------------
z_max = 1e10               # 足够大的最大红移，模拟初始奇点
num_z = 50000              # 积分点数，确保精度
z_array = np.logspace(-4, np.log10(z_max), num_z)  # 对数间隔
# 添加 z=0 点
z_array = np.sort(np.concatenate([[0.0], z_array]))

# ------------------------- Picard 迭代 -------------------------
# 初始猜测：常数 H(z) = H0_bare
H_guess = np.full_like(z_array, H0_bare)

def compute_tau(z, H):
    """计算 lookback time tau(z) = ∫0^z dz' / [(1+z') H(z')]"""
    integrand = 1.0 / ((1.0 + z) * H)
    # 使用累积梯形积分
    tau = cumulative_trapezoid(integrand, z, initial=0.0)
    return tau

max_iter = 20
for it in range(max_iter):
    tau = compute_tau(z_array, H_guess)
    H_new = 1.0 / (T_now - tau)
    # 高红移（z>10）采用解析延拓 H(z) ∝ (1+z)^2
    high_z = z_array > 10.0
    # 在 z=10 处匹配
    H10 = H_new[z_array >= 10.0][0] if np.any(z_array >= 10.0) else H0_bare
    H_new[high_z] = H10 * ((1.0 + z_array[high_z]) / 11.0)**2
    
    diff = np.max(np.abs(H_new - H_guess))
    H_guess = H_new
    if diff < 1e-6:
        print(f"Picard 迭代收敛，步数: {it+1}, 最大差异: {diff:.2e}")
        break
else:
    print("警告：迭代未完全收敛")

H_solution = H_guess

# ------------------------- 计算共形距离 -------------------------
integrand_conf = c / H_solution
chi_array = cumulative_trapezoid(integrand_conf, z_array, initial=0.0)

# 当前粒子视界的共形距离 (z=0 到 z_max)
chi_p = chi_array[-1]   # 单位：米

print(f"\n当前粒子视界共形距离 (理论预测): {chi_p:.3e} m")
print(f"观测值: 4.4e26 m")
print(f"比值 (理论/观测): {chi_p / 4.4e26:.4f}")

# ------------------------- 信息熵估计 -------------------------
I_obs_est = np.log(chi_p / l_P)
print(f"\n唯象信息熵估计 I_obs ≈ ln(Rp/l_P): {I_obs_est:.3f}")

# 对比由公理体系确定的 S_info
alpha_EM = 1.0 / 137.035999084
alpha = (31.0/32.0) * alpha_EM
S_info = 1.0 / alpha
R_max = l_P * np.exp(S_info)
print(f"S_info = 1/α = {S_info:.4f}")
print(f"信息饱和半径 R_max = {R_max:.3e} m")
print(f"Δ = I_obs - S_info = {I_obs_est - S_info:.4f}")