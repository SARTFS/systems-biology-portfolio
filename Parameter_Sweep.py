import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def tat_ltr_model(y,t,k_basal,k_induced,K_half,n,d_tat,d_acrv):
    Tat,ACRV = y

    dTat_dt = k_basal+k_induced*(Tat**n/(K_half**n+Tat**n))-d_tat*Tat
    dACRV_dt = k_induced*(Tat**n/(K_half**n+Tat**n))-d_acrv*ACRV
    return [dTat_dt,dACRV_dt]

y0 = [0.1,0.0]
t = np.linspace(0,50,1000)
params = (0.01,1.0,0.5,2,0.1,0.2)

# 扫描初始Tat浓度
initial_tat_values = [0.01, 0.05, 0.1, 0.3, 0.5, 1]
plt.figure(figsize=(12, 8))

for tat0 in initial_tat_values:
    y0 = [tat0, 0.0]
    sol = odeint(tat_ltr_model, y0, t, args=params)
    plt.plot(t, sol[:, 1], label=f'Tat0={tat0}')

plt.xlabel('Time')
plt.ylabel('ACRV Concentration')
plt.legend()
plt.title('ACRV Activation Threshold: Sensitivity to Initial Tat')
plt.show()
