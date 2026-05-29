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


fixed_params = {
    'k_tat_from_hiv': 0.05, 'k_tat_positive': 1.0, 'K_tat': 0.5, 'n_tat': 2,
    'd_tat': 0.1, 'k_acrv_from_tat': 1.0, 'd_acrv': 0.2, 'd_shrna': 0.05, 'm': 2
}

t_long = np.linspace(0, 300, 3000)

k_shrna_range = np.linspace(0.05, 2.0, 25)
K_silence_range = np.linspace(0.1, 5.0, 25)

peak_acrv = np.zeros((len(K_silence_range), len(k_shrna_range)))
acrv_t300 = np.zeros((len(K_silence_range), len(k_shrna_range)))
selectivity = np.zeros((len(K_silence_range), len(k_shrna_range)))

for i, K_sil in enumerate(K_silence_range):
    for j, k_sh in enumerate(k_shrna_range):
        params = (fixed_params['k_tat_from_hiv'], fixed_params['k_tat_positive'],
                  fixed_params['K_tat'], fixed_params['n_tat'], fixed_params['d_tat'],
                  fixed_params['k_acrv_from_tat'], fixed_params['d_acrv'],
                  k_sh, fixed_params['d_shrna'], K_sil, fixed_params['m'])

        sol_infected = odeint(acrv_hiv_model, [1.0, 0.0, 0.0, 0.0], t_long, args=params)
        sol_uninfected = odeint(acrv_hiv_model, [0.0, 0.0, 0.1, 0.0], t_long, args=params)

        acrv_trace = sol_infected[:, 2]
        peak_acrv[i, j] = np.max(acrv_trace)
        acrv_t300[i, j] = acrv_trace[-1]  # t=300终值

        infected_total = np.trapz(sol_infected[:, 2], t_long)
        uninfected_total = np.trapz(sol_uninfected[:, 2], t_long)
        selectivity[i, j] = infected_total / (uninfected_total + 1e-10)

# 绘图
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

im0 = axes[0].imshow(peak_acrv, aspect='auto', origin='lower',
                     extent=[k_shrna_range[0], k_shrna_range[-1], K_silence_range[0], K_silence_range[-1]],
                     cmap='viridis')
axes[0].set_title('ACRV Peak');
axes[0].set_xlabel('k_shrna');
axes[0].set_ylabel('K_silence')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(np.log10(selectivity), aspect='auto', origin='lower',
                     extent=[k_shrna_range[0], k_shrna_range[-1], K_silence_range[0], K_silence_range[-1]],
                     cmap='RdYlGn')
axes[1].set_title('Selectivity (log10)');
axes[1].set_xlabel('k_shrna');
axes[1].set_ylabel('K_silence')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(acrv_t300, aspect='auto', origin='lower',
                     extent=[k_shrna_range[0], k_shrna_range[-1], K_silence_range[0], K_silence_range[-1]],
                     cmap='plasma')
axes[2].set_title('ACRV at t=300');
axes[2].set_xlabel('k_shrna');
axes[2].set_ylabel('K_silence')
plt.colorbar(im2, ax=axes[2])

# 关键：真黄金窗口 = 选择性>100 + 峰值>2 + t300<0.1
true_gold = (selectivity > 100) & (peak_acrv > 2.0) & (acrv_t300 < 0.1)
im3 = axes[3].imshow(true_gold, aspect='auto', origin='lower',
                     extent=[k_shrna_range[0], k_shrna_range[-1], K_silence_range[0], K_silence_range[-1]],
                     cmap='Greens')
axes[3].set_title('TRUE Goldilocks Window\n(Selectivity>100, Peak>2, t300<0.1)')
axes[3].set_xlabel('k_shrna');
axes[3].set_ylabel('K_silence')

plt.tight_layout()
plt.savefig('acrv_true_design_space.png', dpi=150)
plt.show()

# 输出结果
if np.any(true_gold):
    gold_idx = np.where(true_gold)
    print(f" 真黄金窗口存在,参数组合数: {len(gold_idx[0])}")
    print(f"   k_shrna范围: {k_shrna_range[gold_idx[1]].min():.3f} - {k_shrna_range[gold_idx[1]].max():.3f}")
    print(f"   K_silence范围: {K_silence_range[gold_idx[0]].min():.3f} - {K_silence_range[gold_idx[0]].max():.3f}")

    # 找到最佳平衡点
    best_idx = None
    best_score = 0
    for gi, gj in zip(gold_idx[0], gold_idx[1]):
        score = selectivity[gi, gj] * peak_acrv[gi, gj] / (acrv_t300[gi, gj] + 0.01)
        if score > best_score:
            best_score = score
            best_idx = (gi, gj)

    if best_idx:
        print(f"\n 最佳平衡点:")
        print(f"   k_shrna={k_shrna_range[best_idx[1]]:.3f}, K_silence={K_silence_range[best_idx[0]]:.3f}")
        print(f"   选择性={selectivity[best_idx]:.1f}x, 峰值={peak_acrv[best_idx]:.2f}, t300={acrv_t300[best_idx]:.4f}")
else:
    print(" 真黄金窗口不存在（没有参数能同时满足三个条件）")
    print("   建议：降低清除阈值（t300<0.5）或放宽选择性要求")