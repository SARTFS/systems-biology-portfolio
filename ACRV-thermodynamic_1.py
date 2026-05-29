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
    dTat_dt = (k_tat_from_hiv * Provirus + k_tat_positive * (
                Tat ** n_tat / (K_tat ** n_tat + Tat ** n_tat))) * inhibition - d_tat * Tat

    # ACRV产生：严格依赖Tat
    if Tat > 0.001:
        dACRV_dt = k_acrv_from_tat * (Tat ** n_tat / (K_tat ** n_tat + Tat ** n_tat)) - d_acrv * ACRV
    else:
        dACRV_dt = -d_acrv * ACRV

    # shRNA：由ACRV基因组表达
    dshRNA_dt = k_shrna * ACRV - d_shrna * shRNA

    # Provirus：基本不变
    dProvirus_dt = 0.0

    return [dProvirus_dt, dTat_dt, dACRV_dt, dshRNA_dt]


# ============================================
# 最优参数组合（来自热力图分析）
# ============================================
k_shrna_opt = 0.05  # shRNA表达速率
K_silence_opt = 5.0  # shRNA半数抑制浓度

params_optimal = (
    0.05,  # k_tat_from_hiv
    1.0,  # k_tat_positive
    0.5,  # K_tat
    2,  # n_tat
    0.1,  # d_tat
    1.0,  # k_acrv_from_tat
    0.2,  # d_acrv
    k_shrna_opt,
    0.05,  # d_shrna
    K_silence_opt,
    2  # m
)

# ============================================
# 对比参数组合（黄金窗口内，较强shRNA）
# ============================================
params_strong_shrna = (
    0.05, 1.0, 0.5, 2, 0.1, 1.0, 0.2,
    1.0,  # k_shrna = 1.0（强表达）
    0.05, 1.0, 2  # K_silence = 1.0（强抑制）
)

# ============================================
# 模拟设置
# ============================================
t_short = np.linspace(0, 100, 1000)  # 原始模拟时长
t_long = np.linspace(0, 300, 3000)  # 延长验证时长

y0_infected = [1.0, 0.0, 0.0, 0.0]  # 感染细胞
y0_uninfected = [0.0, 0.0, 0.1, 0.0]  # 未感染细胞

# ============================================
# 运行模拟：最优参数
# ============================================
sol_short_opt = odeint(acrv_hiv_model, y0_infected, t_short, args=params_optimal)
sol_long_opt = odeint(acrv_hiv_model, y0_infected, t_long, args=params_optimal)

sol_short_uninf = odeint(acrv_hiv_model, y0_uninfected, t_short, args=params_optimal)
sol_long_uninf = odeint(acrv_hiv_model, y0_uninfected, t_long, args=params_optimal)

# ============================================
# 运行模拟：对比参数（强shRNA）
# ============================================
sol_long_strong = odeint(acrv_hiv_model, y0_infected, t_long, args=params_strong_shrna)

# ============================================
# 绘图
# ============================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 第一行：最优参数，感染细胞
axes[0, 0].plot(t_short, sol_short_opt[:, 2], 'b-', label='t=0-100')
axes[0, 0].plot(t_long, sol_long_opt[:, 2], 'b--', alpha=0.7, label='t=0-300')
axes[0, 0].set_title(f'Optimal: Infected Cell\nk_shrna={k_shrna_opt}, K_silence={K_silence_opt}')
axes[0, 0].set_xlabel('Time');
axes[0, 0].set_ylabel('ACRV')
axes[0, 0].legend()

axes[0, 1].plot(t_short, sol_short_opt[:, 1], 'r-', label='t=0-100')
axes[0, 1].plot(t_long, sol_long_opt[:, 1], 'r--', alpha=0.7, label='t=0-300')
axes[0, 1].set_title('Tat (Optimal)')
axes[0, 1].set_xlabel('Time');
axes[0, 1].set_ylabel('Tat')
axes[0, 1].legend()

axes[0, 2].plot(t_short, sol_short_opt[:, 3], 'g-', label='t=0-100')
axes[0, 2].plot(t_long, sol_long_opt[:, 3], 'g--', alpha=0.7, label='t=0-300')
axes[0, 2].set_title('shRNA (Optimal)')
axes[0, 2].set_xlabel('Time');
axes[0, 2].set_ylabel('shRNA')
axes[0, 2].legend()

# 第二行：对比
axes[1, 0].plot(t_long, sol_long_opt[:, 2], 'b-', label='Optimal (k_sh=0.05)')
axes[1, 0].plot(t_long, sol_long_strong[:, 2], 'r-', label='Strong shRNA (k_sh=1.0)')
axes[1, 0].set_title('ACRV: Optimal vs Strong shRNA')
axes[1, 0].set_xlabel('Time');
axes[1, 0].set_ylabel('ACRV')
axes[1, 0].legend()

axes[1, 1].plot(t_long, sol_long_opt[:, 2], 'b-', label='Infected')
axes[1, 1].plot(t_long, sol_long_uninf[:, 2], 'b--', alpha=0.5, label='Uninfected')
axes[1, 1].set_title('Selectivity Validation (t=300)')
axes[1, 1].set_xlabel('Time');
axes[1, 1].set_ylabel('ACRV')
axes[1, 1].legend()

# 累积暴露对比
cum_inf_opt = np.cumsum(sol_long_opt[:, 2]) * (t_long[1] - t_long[0])
cum_uninf_opt = np.cumsum(sol_long_uninf[:, 2]) * (t_long[1] - t_long[0])
axes[1, 2].semilogy(t_long, cum_inf_opt, 'b-', label='Infected')
axes[1, 2].semilogy(t_long, cum_uninf_opt, 'b--', alpha=0.5, label='Uninfected')
axes[1, 2].set_title('Cumulative Exposure (log scale)')
axes[1, 2].set_xlabel('Time');
axes[1, 2].set_ylabel('Cumulative ACRV')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('acrv_extended_validation_t300.png', dpi=150)
plt.show()

# ============================================
# 数值输出
# ============================================
print("=" * 60)
print("EXTENDED TIME COURSE VALIDATION (t=300)")
print("=" * 60)

print(f"\n【最优参数】k_shrna={k_shrna_opt}, K_silence={K_silence_opt}")
print(f"  t=100:  ACRV={sol_long_opt[1000, 2]:.6f}, Tat={sol_long_opt[1000, 1]:.6f}, shRNA={sol_long_opt[1000, 3]:.6f}")
print(f"  t=200:  ACRV={sol_long_opt[2000, 2]:.6f}, Tat={sol_long_opt[2000, 1]:.6f}, shRNA={sol_long_opt[2000, 3]:.6f}")
print(
    f"  t=300:  ACRV={sol_long_opt[3000 - 1, 2]:.6f}, Tat={sol_long_opt[3000 - 1, 1]:.6f}, shRNA={sol_long_opt[3000 - 1, 3]:.6f}")

print(f"\n【强shRNA对比】k_shrna=1.0, K_silence=1.0")
print(f"  t=100:  ACRV={sol_long_strong[1000, 2]:.6f}")
print(f"  t=200:  ACRV={sol_long_strong[2000, 2]:.6f}")
print(f"  t=300:  ACRV={sol_long_strong[3000 - 1, 2]:.6f}")

print(f"\n【未感染细胞】最优参数")
print(f"  t=100:  ACRV={sol_long_uninf[1000, 2]:.6f}")
print(f"  t=300:  ACRV={sol_long_uninf[3000 - 1, 2]:.6f}")

# 计算延长模拟后的选择性指数
infected_total_long = np.trapz(sol_long_opt[:, 2], t_long)
uninfected_total_long = np.trapz(sol_long_uninf[:, 2], t_long)
selectivity_long = infected_total_long / (uninfected_total_long + 1e-10)

print(f"\n【选择性指数对比】")
print(f"  t=100模拟: 选择性指数 = 907.3x (之前结果)")
print(f"  t=300模拟: 选择性指数 = {selectivity_long:.1f}x")

# 判断ACRV是否"归零"（低于阈值）
threshold_clearance = 0.01  # 认为<0.01为"清除"
acrv_final = sol_long_opt[3000 - 1, 2]
if acrv_final < threshold_clearance:
    print(f"\n ACRV在t=300时已清除 (ACRV={acrv_final:.6f} < {threshold_clearance})")
else:
    print(f"\n ACRV在t=300时仍未完全清除 (ACRV={acrv_final:.6f} >= {threshold_clearance})")
    print(f"   建议：可能需要更强的shRNA或更长的模拟时间")