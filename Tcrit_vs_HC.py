from Simulator import Simulator
import numpy as np
from scipy import misc 
import matplotlib.pyplot as plt
#from pylab import *

A = misc.imread('Terrain.png')[200:400,200:400].T
A = A/np.max(A)
A = np.zeros((100,100))
nx=A.shape[0];ny=A.shape[0];nt=200;                         # Fundamental simulation parameters
T = np.zeros((nx,ny))                                    # Initialize T field
T[int(nx/2):int(nx/2+2),int(ny/2):int(ny/2+3)] = 1000      # Increase temperature in the middle of the grid
F = np.random.normal(loc=500, scale=0, size=(nx,ny))

heatContentX = np.linspace(0,150,21)#[20 + i for i in range(10)]
TcritX = np.linspace(1000,0,21) #[150 + 10*i for i in range(10)]

results = np.zeros((len(TcritX), len(heatContentX)))

for i in range(len(TcritX)):
    for j in range(len(heatContentX)):
        for k in range(1):
            F = np.random.normal(loc=500, scale=0, size=(nx,ny))
            sim = Simulator(nx,ny,nt,T=T.copy(),A=A.copy(),F=F.copy(),H=np.zeros((nx,ny)),
                            Tcrit=TcritX[i],
                            burningRate=5,
                            heatContent=heatContentX[j],
                            planarDiffusivity=0.8,
                            atmosphericDiffusivity=.56,
                            slopeContribution=1)        # Initialize fields and parameters
            sim.Run(animStep=0)                                   # Perform the simulation
            results[i,j] += sim.Metrics()['burntFuel']
            print(sim.Metrics()['burntFuel'])
            #print(sim.Metrics()['burntStep'][-10:])

cmap = plt.cm.get_cmap('inferno', 11)

img = plt.imshow(results/1, cmap=cmap, origin='lower', interpolation='bicubic')

plt.colorbar(img)
plt.clim(0, 1)
skip = 3
plt.yticks(range(0,len(TcritX), skip)      , TcritX[::skip])
plt.xticks(range(0,len(heatContentX), skip), heatContentX[::skip])
plt.ylabel('$T_{crit}$')
plt.xlabel('$\lambda_{HC}$')
plt.title('Pencentage of area burnt in 200 steps')

plt.show()
#sim.Show()                                  # Visualize the results
