from Simulator import Simulator
import numpy as np
from scipy import misc 
import matplotlib.pyplot as plt
#from pylab import *

A = np.zeros((50,50))
nx=A.shape[0];ny=A.shape[0];nt=10000;                         # Fundamental simulation parameters
T = np.zeros((nx,ny))                                    # Initialize T field
T[int(nx/2):int(nx/2+2),int(ny/2):int(ny/2+2)] = 500      # Increase temperature in the middle of the grid
F = np.random.normal(loc=500, scale=0, size=(nx,ny))
F[0,:] = 0; F[-1,:] = 0;F[:,-1] = 0; F[:,0] = 0

heatContentX = np.linspace(100,600,7)
TcritX = np.linspace(100,30,7)

results = np.zeros((len(TcritX), len(heatContentX)))
dead = np.zeros((len(TcritX), len(heatContentX)), dtype=bool)

for i in range(len(TcritX)):
    for j in range(len(heatContentX)):
        for k in range(1):
            F = np.random.normal(loc=500, scale=0, size=(nx,ny))
            sim = Simulator(nx,ny,nt,T=T.copy(),A=A.copy(),F=F.copy(),H=np.zeros((nx,ny)), dt=0.01,
                            Tcrit=TcritX[i],
                            burningRate=0.02,
                            heatContent=heatContentX[j],
                            planarDiffusivity=0.005,
                            atmosphericDiffusivity=0.004,
                            slopeContribution=1)        # Initialize fields and parameters
            sim.Run(animStep=0)                                   # Perform the simulation
            results[i,j] += sim.Metrics()['totalBurnt']
            dead[i,j] = sim.Metrics()['burning']
            #print(sim.Metrics()['burntFuel'])
            #print(sim.Metrics()['dead'])
            #print(sim.Metrics()['burntStep'][-10:])
            print((i*len(heatContentX) + j + 1)/(len(heatContentX) * len(TcritX))*100, '%', end='\r')

print(dead)

cmap = 'inferno'
img = plt.imshow(results/1, cmap=cmap, origin='lower', interpolation=None)

plt.show()

cmap = plt.cm.get_cmap('inferno', 11)
img = plt.imshow(results/1, cmap=cmap, origin='lower', interpolation='bicubic')

plt.colorbar(img)
#plt.clim(0, 1)
skip = 3
plt.yticks(range(0,len(TcritX), skip)      , TcritX[::skip])
plt.xticks(range(0,len(heatContentX), skip), heatContentX[::skip])
plt.ylabel('$T_{crit}$')
plt.xlabel('$\lambda_{HC}$')
plt.title('Fraction of area burnt in {} steps'.format(nt))

plt.savefig('plots/Tcrit_vs_HC.png')

plt.show()
#sim.Show()                                  # Visualize the results
