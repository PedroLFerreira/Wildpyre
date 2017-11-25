from Simulator import Simulator
import numpy as np
from scipy import misc 
import matplotlib.pyplot as plt
import numpy.core.defchararray as npstr
#from pylab import *

A = np.zeros((50,50))
nx=A.shape[0];ny=A.shape[0];nt=10000;                         # Fundamental simulation parameters
T = np.zeros((nx,ny))                                    # Initialize T field
T[int(nx/2):int(nx/2+2),int(ny/2):int(ny/2+3)] = 1000      # Increase temperature in the middle of the grid
F = np.random.normal(loc=500, scale=0, size=(nx,ny))

planarX = np.logspace(-2,0,21)#[20 + i for i in range(10)]
atmosX = np.logspace(-1.6,-1,21) #[150 + 10*i for i in range(10)]

results = np.zeros((len(atmosX), len(planarX)))
dead = np.zeros((len(atmosX), len(planarX)), dtype=bool)

for i in range(len(atmosX)):
    for j in range(len(planarX)):
        for k in range(1):
            F = np.random.normal(loc=500, scale=0, size=(nx,ny))
            sim = Simulator(nx,ny,nt,T=T.copy(),A=A.copy(),F=F.copy(),H=np.zeros((nx,ny)), dt=0.01,
                            Tcrit=100,
                            burningRate=0.02,
                            heatContent=90,
                            planarDiffusivity=planarX[j],
                            atmosphericDiffusivity=atmosX[i],
                            slopeContribution=1)        # Initialize fields and parameters
            sim.Run(animStep=0)                                   # Perform the simulation
            results[i,j] += sim.Metrics()['totalBurnt']
            #print(sim.Metrics()['burntFuel'])
            #print(sim.Metrics()['dead'])
            #print(sim.Metrics()['burntStep'][-10:])

            print((i*len(planarX) + j + 1)/(len(planarX) * len(atmosX))*100, '%', end='\r')

print(dead)

cmap = 'inferno'
img = plt.imshow(results/1, cmap=cmap, origin='lower', interpolation=None)

skip = 3
plt.yticks(range(0,len(atmosX), skip) , npstr.add(npstr.add('$10^{', np.around(np.log10(atmosX[::skip]),decimals=1).astype(str)), '}$'))
plt.xticks(range(0,len(planarX), skip), npstr.add(npstr.add('$10^{', np.around(np.log10(planarX[::skip]),decimals=1).astype(str)), '}$'))
plt.ylabel(r'$\alpha_{atm}$')
plt.xlabel(r'$\alpha$')
plt.title('Fraction of area burnt in {} steps'.format(nt))

plt.savefig('plots/planar_vs_atmos_nointerpol.png')

plt.show()

cmap = plt.cm.get_cmap('inferno', 11)
img = plt.imshow(results/1, cmap=cmap, origin='lower', interpolation='bicubic')

plt.colorbar(img)
#plt.clim(0, 1)
skip = 3
plt.yticks(range(0,len(atmosX), skip) , npstr.add(npstr.add('$10^{', np.around(np.log10(atmosX[::skip]),decimals=1).astype(str)), '}$'))
plt.xticks(range(0,len(planarX), skip), npstr.add(npstr.add('$10^{', np.around(np.log10(planarX[::skip]),decimals=1).astype(str)), '}$'))
plt.ylabel(r'$\alpha_{atm}$')
plt.xlabel(r'$\alpha$')
plt.title('Fraction of area burnt in {} steps'.format(nt))

plt.savefig('plots/planar_vs_atmos.png')

plt.show()
#sim.Show()                                  # Visualize the results
