from Simulator import Simulator
import numpy as np
from scipy import misc 
import matplotlib.pyplot as plt

A = misc.imread('Terrain.png')[200:400,200:400].T
A = A/np.max(A)
A = np.zeros((50,50))
nx=A.shape[0];ny=A.shape[0];nt=100000;                         # Fundamental simulation parameters
T = np.zeros((nx,ny))                                    # Initialize T field
#T[int(nx/2):int(nx/2+2),int(ny/2):int(ny/2+3)] = 1000      # Increase temperature in the middle of the grid
T[nx//2, ny//2] = 1000
F = np.random.normal(loc=500, scale=0, size=(nx,ny))
F = np.maximum(F, 0)

# sim = Simulator(nx,ny,nt,T=T,A=A,F=F,dt=1,
#                 Tcrit=150,
#                 burningRate=0.005,
#                 heatContent=5,
#                 planarDiffusivity=0.005,
#                 atmosphericDiffusivity=0.004,
#                 slopeContribution=1)        # Initialize fields and parameters
# sim.Run(animStep=20)                                   # Perform the simulation
# sim.Show()                                  # Visualize the results

#resolution = 100
#resultados = np.zeros(resolution)
#alphas = np.linspace(0.001, 0.0025, resolution)

#for i, alpha in enumerate(alphas):
sim = Simulator(nx,ny,nt,T=T.copy(),A=A.copy(),F=F.copy(),dt=0.1,
                Tcrit=100,
                burningRate=0.02,
                heatContent=90,
                planarDiffusivity=0.1,
                atmosphericDiffusivity=0.01,
                slopeContribution=1)        # Initialize fields and parameters
sim.Run(animStep=100)                                   # Perform the simulation

#resultados[i] = sim.Metrics()['elapsedTime']

print(sim.Metrics()['totalBurnt'])
print(sim.Metrics()['burntStep'])

print(np.sum(sim.H[nx//2,] > 0)//2)
#plt.show()

#plt.plot(alphas, resultados, '-o')
#plt.show()