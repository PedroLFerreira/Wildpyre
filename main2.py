from Simulator import Simulator
import numpy as np
from scipy import misc 
import matplotlib.pyplot as plt

A = misc.imread('Terrain.png')[200:400,200:400].T
A = A/np.max(A)
nx=A.shape[0];ny=A.shape[0];nt=500;                         # Fundamental simulation parameters
T = np.zeros((nx,ny))                                    # Initialize T field
T[int(nx/2):int(nx/2+2),int(ny/2):int(ny/2+3)] = 1000      # Increase temperature in the middle of the grid
F = np.random.normal(loc=500, scale=120, size=(nx,ny))

burning = np.zeros(nt-1)
dead = []

for _ in range(1):
    F = np.random.normal(loc=500, scale=120, size=(nx,ny))
    sim = Simulator(nx,ny,nt,T=T.copy(),A=np.zeros_like(A),F=F.copy(),H=np.zeros((nx,ny)),
                    Tcrit=200,
                    burningRate=5,
                    heatContent=25,
                    planarDiffusivity=1.2,
                    atmosphericDiffusivity=.56,
                    slopeContribution=1)        # Initialize fields and parameters
    sim.Run(animStep=50)                                   # Perform the simulation
    print(sim.Metrics()['burntFuel'])
    burning += sim.Metrics()['burntStep']
    print(sim.Metrics()['burntStep'][-10:])
    dead.append((sim.Metrics()['burntStep'] == 0.).any())

print(dead)
plt.plot(burning)
plt.show()
#sim.Show()                                  # Visualize the results
