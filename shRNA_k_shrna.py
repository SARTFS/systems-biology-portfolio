import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def acrv_with_shrna(y, t, k_basal, k_induced, K_half, n,
                    d_tat, d_acrv, k_shrna, d_shrna, K_silence, m):
    Tat, ACRV, shRNA = y

    # shRNA抑制Tat产生
    inhibition = 1.0 / (1.0 + (shRNA / K_silence) ** m)

    dTat_dt = (k_basal + k_induced * (Tat ** n / (K_half ** n + Tat ** n))) * inhibition - d_tat * Tat
    dACRV_dt = k_induced * (Tat ** n / (K_half ** n + Tat ** n)) - d_acrv * ACRV
    dshRNA_dt = k_shrna * ACRV - d_shrna * shRNA

    return [dTat_dt, dACRV_dt, dshRNA_dt]


t = np.linspace(0, 100, 1000)

# 基线参数
base_params = {
    'k_basal': 0.01, 'k_induced': 1.0, 'K_half': 0.5, 'n': 2,
    'd_tat': 0.1, 'd_acrv': 0.2, 'd_shrna': 0.05, 'K_silence': 1.0, 'm': 2
}

# 扫描shRNA表达强度（核心参数）
k_shrna_values = [0.05, 0.1, 0.3, 0.5, 1.0]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for k_sh in k_shrna_values:
    params = (base_params['k_basal'], base_params['k_induced'], base_params['K_half'],
              base_params['n'], base_params['d_tat'], base_params['d_acrv'],
              k_sh, base_params['d_shrna'], base_params['K_silence'], base_params['m'])

    y0 = [0.1, 0.0, 0.0]  # 感染细胞：Tat=0.1
    sol = odeint(acrv_with_shrna, y0, t, args=params)

    axes[0, 0].plot(t, sol[:, 0], label=f'k_sh={k_sh}')
    axes[0, 1].plot(t, sol[:, 1], label=f'k_sh={k_sh}')
    axes[1, 0].plot(t, sol[:, 2], label=f'k_sh={k_sh}')

    # 计算Tat产生效率
    inhibition = 1.0 / (1.0 + (sol[:, 2] / base_params['K_silence']) ** base_params['m'])
    axes[1, 1].plot(t, inhibition, label=f'k_sh={k_sh}')

axes[0, 0].set_title('Tat');
axes[0, 0].set_xlabel('Time');
axes[0, 0].legend()
axes[0, 1].set_title('ACRV');
axes[0, 1].set_xlabel('Time');
axes[0, 1].legend()
axes[1, 0].set_title('shRNA');
axes[1, 0].set_xlabel('Time');
axes[1, 0].legend()
axes[1, 1].set_title('Tat Production Efficiency');
axes[1, 1].set_xlabel('Time');
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('acrv_shrna_sweep.png', dpi=150)
plt.show()
