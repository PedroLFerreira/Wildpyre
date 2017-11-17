import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,1000,10000)
print(x)
y = [max(0,1-1/i) for i in x]
print(y)

plt.plot(x,y)
plt.xlabel(r'$T^t_{x,y}/T_{crit}$')
plt.ylabel(r'$1-T_{crit} / T^t_{x,y}$')
plt.xscale('log')

plt.show()