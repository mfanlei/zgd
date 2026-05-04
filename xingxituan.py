import numpy as np

# ==================== 星系团叠加效应验证 ====================
# 参数设置
N = 1000                      # 星系数量
M_min, M_max = 1e9, 1e11      # 质量范围 (太阳质量)
M_star, alpha = 5e10, -1.25   # Schechter 函数参数 (取自 Broadhurst et al. 2005)

def schechter_sample(N, M_star, alpha, M_min, M_max):
    """从 Schechter 质量函数抽取样本"""
    log_cand = np.random.uniform(np.log10(M_min), np.log10(M_max), N*50)
    M_cand = 10**log_cand
    weights = (M_cand/M_star)**alpha * np.exp(-M_cand/M_star)
    weights /= np.sum(weights)
    return M_cand[np.random.choice(len(M_cand), size=N, replace=False, p=weights)]

def calc_R(masses, weights=None):
    """计算 R = sum sqrt(M_i) / sqrt(total mass)"""
    w = np.ones_like(masses) if weights is None else weights
    return np.sum(w * np.sqrt(masses)) / np.sqrt(np.sum(w * masses))

# 主计算
np.random.seed(42)                       # 固定随机种子以保证可重复性
masses = schechter_sample(N, M_star, alpha, M_min, M_max)
R_full = calc_R(masses)
top20 = np.sort(masses)[-20:]             # 取前 20 个最大质量星系
R_top20 = calc_R(top20)

print(f"完整分布 R = {R_full:.2f}")
print(f"前20星系 R = {R_top20:.2f}")
print(f"总质量 = {np.sum(masses):.3e} M☉")
print(f"质量变异系数 CV = {np.std(masses)/np.mean(masses):.2f}")