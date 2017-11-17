from Simulator import Simulator
import numpy as np
from scipy import misc 
import matplotlib.pyplot as plt
import numpy.core.defchararray as npstr
#from pylab import *

A = misc.imread('Terrain.png')[200:400,200:400].T
A = A/np.max(A)
A = np.zeros((100,100))
nx=A.shape[0];ny=A.shape[0];nt=2000;                         # Fundamental simulation parameters
T = np.zeros((nx,ny))                                    # Initialize T field
T[int(nx/2):int(nx/2+2),int(ny/2):int(ny/2+3)] = 1000      # Increase temperature in the middle of the grid
F = np.random.normal(loc=500, scale=0, size=(nx,ny))
F[0,:] = 0; F[-1,:] = 0;F[:,-1] = 0; F[:,0] = 0

burnX = np.linspace(0.001,0.061,11)#[20 + i for i in range(10)]
atmosX = np.logspace(-4,-1.5,11) #[150 + 10*i for i in range(10)]

results = np.zeros((len(atmosX), len(burnX)))
dead = np.zeros((len(atmosX), len(burnX)), dtype=bool)

for i in range(len(atmosX)):
    for j in range(len(burnX)):
        for k in range(1):
            F = np.random.normal(loc=500, scale=0, size=(nx,ny))
            sim = Simulator(nx,ny,nt,T=T.copy(),A=A.copy(),F=F.copy(),H=np.zeros((nx,ny)), dt=1,
                            Tcrit=150,
                            burningRate=burnX[j],
                            heatContent=5,
                            planarDiffusivity=0.005,
                            atmosphericDiffusivity=atmosX[i],
                            slopeContribution=1)        # Initialize fields and parameters
            sim.Run(animStep=0)                                   # Perform the simulation
            results[i,j] += sim.Metrics()['burntFuel']
            dead[i,j] = sim.Metrics()['dead']
            #print(sim.Metrics()['burntFuel'])
            #print(sim.Metrics()['dead'])
            #print(sim.Metrics()['burntStep'][-10:])
            print((i*len(burnX) + j + 1)/(len(burnX) * len(atmosX))*100, '%', end='\r')

print(dead)

cmap = 'inferno'
img = plt.imshow(results/1, cmap=cmap, origin='lower', interpolation=None)

plt.show()

cmap = plt.cm.get_cmap('inferno', 11)
img = plt.imshow(results/1, cmap=cmap, origin='lower', interpolation='bicubic')

plt.colorbar(img)
plt.clim(0, 1)
skip = 2
plt.xticks(range(0,len(burnX), skip)      , burnX[::skip])
plt.yticks(range(0,len(atmosX), skip) , npstr.add(npstr.add('$10^{', np.log10(atmosX[::skip]).astype(str)), '}$'))
plt.xlabel('$\lambda_{BR}$')
plt.ylabel(r'$\alpha_{atm}$')
plt.title('Fraction of area burnt in {} steps'.format(nt))

plt.savefig('plots/BR_vs_atmos.png')

plt.show()
#sim.Show()                                  # Visualize the results
