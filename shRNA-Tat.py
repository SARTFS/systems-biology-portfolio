import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def tat_ltr_model_with_shrna(y, t, k_basal, k_induced, K_half, n,
                             d_tat, d_acrv, k_shrna, d_shrna,
                             K_silence, m):
    Tat, ACRV, shRNA = y

    # shRNA对Tat产生的抑制（Hill函数，m为抑制系数）
    inhibition = 1.0 / (1.0 + (shRNA / K_silence) ** m)

    # Tat动力学：基础表达 + 诱导表达（受shRNA抑制） - 降解
    dTat_dt = (k_basal + k_induced * (Tat ** n / (K_half ** n + Tat ** n))) * inhibition - d_tat * Tat

    # ACRV动力学：与Tat共享LTR驱动 - 降解
    dACRV_dt = k_induced * (Tat ** n / (K_half ** n + Tat ** n)) - d_acrv * ACRV

    # shRNA动力学：由ACRV基因组表达 - 降解
    dshRNA_dt = k_shrna * ACRV - d_shrna * shRNA

    return [dTat_dt, dACRV_dt, dshRNA_dt]


# 时间轴
t = np.linspace(0, 100, 1000)

# 参数（在基线参数上扩展）
k_basal = 0.01
k_induced = 1.0
K_half = 0.5
n = 2
d_tat = 0.1
d_acrv = 0.2

# shRNA相关参数（这是你要探索的关键）
k_shrna = 0.3  # shRNA表达速率（越高，抑制越强）
d_shrna = 0.05  # shRNA降解速率
K_silence = 1.0  # 半数抑制浓度（shRNA达到此浓度时，Tat产生被抑制50%）
m = 2  # 抑制Hill系数

params = (k_basal, k_induced, K_half, n, d_tat, d_acrv, k_shrna, d_shrna, K_silence, m)

# 初始条件：Tat=0.1（感染细胞），ACRV=0，shRNA=0
y0 = [0.1, 0.0, 0.0]

sol = odeint(tat_ltr_model_with_shrna, y0, t, args=params)

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(t, sol[:, 0], 'r-', label='Tat')
axes[0, 0].set_title('Tat (Positive Feedback Core)')
axes[0, 0].set_xlabel('Time');
axes[0, 0].set_ylabel('Concentration')
axes[0, 0].legend()

axes[0, 1].plot(t, sol[:, 1], 'b-', label='ACRV')
axes[0, 1].set_title('ACRV (Viral Vector)')
axes[0, 1].set_xlabel('Time');
axes[0, 1].set_ylabel('Concentration')
axes[0, 1].legend()

axes[1, 0].plot(t, sol[:, 2], 'g-', label='shRNA')
axes[1, 0].set_title('shRNA (Gene Silencer)')
axes[1, 0].set_xlabel('Time');
axes[1, 0].set_ylabel('Concentration')
axes[1, 0].legend()

# 关键图：shRNA对Tat产生的抑制效率随时间变化
inhibition_over_time = 1.0 / (1.0 + (sol[:, 2] / K_silence) ** m)
axes[1, 1].plot(t, inhibition_over_time, 'purple', label='Tat Production Efficiency')
axes[1, 1].set_title('Tat Production Efficiency (1.0 = Full, 0.0 = Blocked)')
axes[1, 1].set_xlabel('Time');
axes[1, 1].set_ylabel('Relative Efficiency')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('tat_ltr_with_shrna_feedback.png', dpi=150)
plt.show()

# 打印关键稳态值
print(f"Steady State: Tat={sol[-1, 0]:.3f}, ACRV={sol[-1, 1]:.3f}, shRNA={sol[-1, 2]:.3f}")
print(f"Final Tat Production Efficiency: {inhibition_over_time[-1]:.3f}")
