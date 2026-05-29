import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def acrv_hiv_model(y, t, k_tat_from_hiv, k_tat_positive, K_tat, n_tat,
                   d_tat, k_acrv_from_tat, d_acrv, k_shrna, d_shrna,
                   K_silence, m):
    """
    四变量模型：
    y[0] = Provirus (HIV前病毒DNA，近似常数)
    y[1] = Tat
    y[2] = ACRV
    y[3] = shRNA
    """
    Provirus, Tat, ACRV, shRNA = y

    # shRNA对Tat产生的抑制
    inhibition = 1.0 / (1.0 + (shRNA / K_silence) ** m)

    # Tat产生：来自HIV原病毒的基础转录 + Tat自我正反馈 - 降解
    # Provirus近似常数，作为"背景驱动源"
    dTat_dt = (k_tat_from_hiv * Provirus + k_tat_positive * (
                Tat ** n_tat / (K_tat ** n_tat + Tat ** n_tat))) * inhibition - d_tat * Tat

    # ACRV产生：严格依赖Tat（无Tat则无ACRV）
    if Tat > 0.001:  # 数值稳定性
        dACRV_dt = k_acrv_from_tat * (Tat ** n_tat / (K_tat ** n_tat + Tat ** n_tat)) - d_acrv * ACRV
    else:
        dACRV_dt = -d_acrv * ACRV  # 只有衰减，无产生

    # shRNA：由ACRV基因组表达
    dshRNA_dt = k_shrna * ACRV - d_shrna * shRNA

    # Provirus：基本不变（shRNA主要沉默RNA，不切割DNA）
    dProvirus_dt = 0.0

    return [dProvirus_dt, dTat_dt, dACRV_dt, dshRNA_dt]


# 时间轴
t = np.linspace(0, 100, 1000)

# 参数
params = (
    0.05,  # k_tat_from_hiv: HIV原病毒产生Tat的基础速率
    1.0,  # k_tat_positive: Tat正反馈强度
    0.5,  # K_tat: Tat半数激活浓度
    2,  # n_tat: Hill系数
    0.1,  # d_tat: Tat降解
    1.0,  # k_acrv_from_tat: ACRV产生速率（与Tat正反馈共享参数）
    0.2,  # d_acrv: ACRV降解
    0.3,  # k_shrna: shRNA表达
    0.05,  # d_shrna: shRNA降解
    1.0,  # K_silence: shRNA半数抑制
    2  # m: 抑制Hill系数
)

# 场景1：HIV感染细胞（Provirus = 1.0）
y0_infected = [1.0, 0.0, 0.0, 0.0]  # 有HIV，初始Tat=0
sol_infected = odeint(acrv_hiv_model, y0_infected, t, args=params)

# 场景2：未感染细胞（Provirus = 0.0，但注入微量ACRV测试泄漏）
y0_uninfected = [0.0, 0.0, 0.1, 0.0]  # 无HIV，但有微量ACRV残留
sol_uninfected = odeint(acrv_hiv_model, y0_uninfected, t, args=params)

# 绘图对比
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(t, sol_infected[:, 1], 'r-', label='Infected')
axes[0, 0].plot(t, sol_uninfected[:, 1], 'r--', alpha=0.5, label='Uninfected')
axes[0, 0].set_title('Tat');
axes[0, 0].legend()

axes[0, 1].plot(t, sol_infected[:, 2], 'b-', label='Infected')
axes[0, 1].plot(t, sol_uninfected[:, 2], 'b--', alpha=0.5, label='Uninfected')
axes[0, 1].set_title('ACRV');
axes[0, 1].legend()

axes[1, 0].plot(t, sol_infected[:, 3], 'g-', label='Infected')
axes[1, 0].plot(t, sol_uninfected[:, 3], 'g--', alpha=0.5, label='Uninfected')
axes[1, 0].set_title('shRNA');
axes[1, 0].legend()

# 关键图：ACRV在感染vs未感染细胞中的累积量
axes[1, 1].plot(t, np.cumsum(sol_infected[:, 2]), 'b-', label='Infected (Total ACRV)')
axes[1, 1].plot(t, np.cumsum(sol_uninfected[:, 2]), 'b--', alpha=0.5, label='Uninfected (Total ACRV)')
axes[1, 1].set_title('Cumulative ACRV Exposure');
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('acrv_hiv_provirus_model.png', dpi=150)
plt.show()

print("Infected cell - Final Tat:", sol_infected[-1, 1])
print("Uninfected cell - Final ACRV:", sol_uninfected[-1, 2])

# 计算累积暴露量（AUC）
infected_total = np.trapz(sol_infected[:, 2], t)
uninfected_total = np.trapz(sol_uninfected[:, 2], t)

selectivity_index = infected_total / (uninfected_total + 1e-10)  # 防止除零

print(f"感染细胞ACRV总暴露: {infected_total:.2f}")
print(f"未感染细胞ACRV总暴露: {uninfected_total:.2f}")
print(f"选择性指数 (感染/未感染): {selectivity_index:.1f}x")

# 可视化选择性指数随时间变化
fig, ax = plt.subplots(figsize=(8, 5))

cum_infected = np.cumsum(sol_infected[:, 2]) * (t[1]-t[0])
cum_uninfected = np.cumsum(sol_uninfected[:, 2]) * (t[1]-t[0])
selectivity_over_time = cum_infected / (cum_uninfected + 1e-10)

ax.semilogy(t, selectivity_over_time, 'k-', linewidth=2)
ax.set_title('ACRV Selectivity Index Over Time (Infected / Uninfected)')
ax.set_xlabel('Time')
ax.set_ylabel('Selectivity Ratio (log scale)')
ax.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100x threshold')
ax.legend()
plt.savefig('acrv_selectivity_index.png', dpi=150)
plt.show()
