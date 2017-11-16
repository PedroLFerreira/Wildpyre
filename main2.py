from Simulator import Simulator
import numpy as np
from scipy import misc 

A = misc.imread('Terrain.png')[200:400,200:400].T
A = A/np.max(A)
nx=A.shape[0];ny=A.shape[0];nt=500;                         # Fundamental simulation parameters
T = np.zeros((nx,ny))                                    # Initialize T field
T[int(nx/2):int(nx/2+2),int(ny/2):int(ny/2+3)] = 1000      # Increase temperature in the middle of the grid
F = np.random.normal(loc=500, scale=120, size=(nx,ny))

for _ in range(5):
    F = np.random.normal(loc=500, scale=120, size=(nx,ny))
    sim = Simulator(nx,ny,nt,T=T.copy(),A=A.copy(),F=F.copy(),H=np.zeros((nx,ny)),
                    Tcrit=200,
                    burningRate=5,
                    heatContent=29,
                    planarDiffusivity=1.2,
                    atmosphericDiffusivity=.56,
                    slopeContribution=1)        # Initialize fields and parameters
    sim.Run(animStep=100)                                   # Perform the simulation
    print(sim.Metrics()['burntFuel'])
#sim.Show()                                  # Visualize the results
