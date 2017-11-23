import matplotlib.pyplot as plt
import numpy as np

x = np.logspace(-1,3,10000)
print(x)
y = [max(0,1-1/i) for i in x]
yarrhenius = [np.exp(-1/i) for i in x]
print(y)

plt.plot(x,y,label=r'$(1-T_{crit} / T^t_{x,y})\lambda_{BR}$')
plt.plot(x,yarrhenius,label='Arrhenius rate\n'+ r'$Ae^{-T_{crit}/T^t_{x,y}}$')
plt.xlabel(r'$T^t_{x,y}/T_{crit}$')
plt.ylabel(r'$\Delta T^t_{x,y}$')
plt.xscale('log')
plt.legend()
plt.show()