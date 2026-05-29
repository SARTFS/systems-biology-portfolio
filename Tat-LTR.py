import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def tat_ltr_model(y,t,k_basal,k_induced,K_half,n,d_tat,d_acrv):
    Tat,ACRV = y

    dTat_dt = k_basal+k_induced*(Tat**n/(K_half**n+Tat**n))-d_tat*Tat
    dACRV_dt = k_induced*(Tat**n/(K_half**n+Tat**n))-d_acrv*ACRV
    return [dTat_dt,dACRV_dt]

y0 = [0.1,0.0]
t = np.linspace(0,50,1000)
params = (0.01,1.0,0.5,2,0.1,0.2)
sol = odeint(tat_ltr_model,y0,t,args = params)

plt.figure(figsize=(10,6))
plt.plot(t,sol[:,0],label='Tat',color='red')
plt.plot(t,sol[:,1],label='ACRV',color='blue')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Tat-LTR Positive Feedback Model')
plt.legend()
plt.grid(True)
plt.show()

