import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def acrv_hiv_model(y, t, k_tat_from_hiv, k_tat_positive, K_tat, n_tat,
                   d_tat, k_acrv_from_tat, d_acrv, k_shrna, d_shrna,
                   K_silence, m):
    Provirus, Tat, ACRV, shRNA = y
    inhibition = 1.0 / (1.0 + (shRNA / K_silence) ** m)
    dTat_dt = (k_tat_from_hiv * Provirus + k_tat_positive * (
                Tat ** n_tat / (K_tat ** n_tat + Tat ** n_tat))) * inhibition - d_tat * Tat
    dACRV_dt = k_acrv_from_tat * (Tat ** n_tat / (K_tat ** n_tat + Tat ** n_tat)) - d_acrv * ACRV
    dshRNA_dt = k_shrna * ACRV - d_shrna * shRNA
    dProvirus_dt = 0.0
    return [dProvirus_dt, dTat_dt, dACRV_dt, dshRNA_dt]


# 固定参数
fixed_params = {
    'k_tat_from_hiv': 0.05, 'k_tat_positive': 1.0, 'K_tat': 0.5, 'n_tat': 2,
    'd_tat': 0.1, 'k_acrv_from_tat': 1.0, 'd_acrv': 0.2, 'd_shrna': 0.05, 'm': 2
}

t = np.linspace(0, 100, 1000)

# 扫描网格
k_shrna_range = np.linspace(0.05, 2.0, 30)
K_silence_range = np.linspace(0.1, 5.0, 30)

# 存储结果
peak_acrv = np.zeros((len(K_silence_range), len(k_shrna_range)))
duration_acrv = np.zeros((len(K_silence_range), len(k_shrna_range)))
selectivity = np.zeros((len(K_silence_range), len(k_shrna_range)))

# 运行扫描
for i, K_sil in enumerate(K_silence_range):
    for j, k_sh in enumerate(k_shrna_range):
        params = (fixed_params['k_tat_from_hiv'], fixed_params['k_tat_positive'],
                  fixed_params['K_tat'], fixed_params['n_tat'], fixed_params['d_tat'],
                  fixed_params['k_acrv_from_tat'], fixed_params['d_acrv'],
                  k_sh, fixed_params['d_shrna'], K_sil, fixed_params['m'])

        # 感染细胞
        sol_infected = odeint(acrv_hiv_model, [1.0, 0.0, 0.0, 0.0], t, args=params)
        # 未感染细胞
        sol_uninfected = odeint(acrv_hiv_model, [0.0, 0.0, 0.1, 0.0], t, args=params)

        acrv_trace = sol_infected[:, 2]

        # 指标1: ACRV峰值
        peak_acrv[i, j] = np.max(acrv_trace)

        # 指标2: 脉冲持续时间（ACRV > 峰值10%的时间）
        threshold = 0.1 * np.max(acrv_trace)
        duration_acrv[i, j] = np.sum(acrv_trace > threshold) * (t[1] - t[0])

        # 指标3: 选择性指数
        infected_total = np.trapz(sol_infected[:, 2], t)
        uninfected_total = np.trapz(sol_uninfected[:, 2], t)
        selectivity[i, j] = infected_total / (uninfected_total + 1e-10)

# 绘图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 子图1: ACRV峰值
im1 = axes[0].imshow(peak_acrv, aspect='auto', origin='lower',
                     extent=[k_shrna_range[0], k_shrna_range[-1],
                             K_silence_range[0], K_silence_range[-1]],
                     cmap='viridis')
axes[0].set_xlabel('k_shrna (shRNA expression rate)')
axes[0].set_ylabel('K_silence (Half-maximal inhibition)')
axes[0].set_title('ACRV Peak Height\n(Therapeutic Efficacy)')
plt.colorbar(im1, ax=axes[0])

# 子图2: 脉冲持续时间
im2 = axes[1].imshow(duration_acrv, aspect='auto', origin='lower',
                     extent=[k_shrna_range[0], k_shrna_range[-1],
                             K_silence_range[0], K_silence_range[-1]],
                     cmap='plasma')
axes[1].set_xlabel('k_shrna (shRNA expression rate)')
axes[1].set_ylabel('K_silence (Half-maximal inhibition)')
axes[1].set_title('ACRV Pulse Duration\n(Safety: shorter is safer)')
plt.colorbar(im2, ax=axes[1])

# 子图3: 选择性指数（对数坐标）
im3 = axes[2].imshow(np.log10(selectivity), aspect='auto', origin='lower',
                     extent=[k_shrna_range[0], k_shrna_range[-1],
                             K_silence_range[0], K_silence_range[-1]],
                     cmap='RdYlGn')
axes[2].set_xlabel('k_shrna (shRNA expression rate)')
axes[2].set_ylabel('K_silence (Half-maximal inhibition)')
axes[2].set_title('Selectivity Index (log10)\n(Infected / Uninfected)')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig('acrv_design_space_heatmap.png', dpi=150)
plt.show()

# 找到最优设计点
optimal_idx = np.unravel_index(np.argmax(selectivity), selectivity.shape)
print(f"最优选择性指数: {selectivity[optimal_idx]:.1f}x")
print(f"  对应参数: k_shrna={k_shrna_range[optimal_idx[1]]:.3f}, K_silence={K_silence_range[optimal_idx[0]]:.3f}")
print(f"  该点ACRV峰值: {peak_acrv[optimal_idx]:.2f}")
print(f"  该点脉冲持续时间: {duration_acrv[optimal_idx]:.1f}")

# 找到"黄金窗口"：选择性>100x 且 峰值>2.0 且 持续时间<40
gold_mask = (selectivity > 100) & (peak_acrv > 2.0) & (duration_acrv < 40)
if np.any(gold_mask):
    gold_indices = np.where(gold_mask)
    print(f"\n黄金窗口参数组合数: {len(gold_indices[0])}")
    print(f"  k_shrna范围: {k_shrna_range[gold_indices[1]].min():.3f} - {k_shrna_range[gold_indices[1]].max():.3f}")
    print(
        f"  K_silence范围: {K_silence_range[gold_indices[0]].min():.3f} - {K_silence_range[gold_indices[0]].max():.3f}")
else:
    print("\n未找到同时满足三个条件的黄金窗口")
