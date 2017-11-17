from Simulator import Simulator
import numpy as np
from scipy import misc 

A = misc.imread('Terrain.png')[200:400,200:400].T
A = A/np.max(A)
A = np.zeros((100,100))
nx=A.shape[0];ny=A.shape[0];nt=2000;                         # Fundamental simulation parameters
T = np.zeros((nx,ny))                                    # Initialize T field
T[int(nx/2):int(nx/2+2),int(ny/2):int(ny/2+3)] = 1000      # Increase temperature in the middle of the grid
F = np.random.normal(loc=500, scale=120, size=(nx,ny))

sim = Simulator(nx,ny,nt,T=T,A=A,F=F,
                Tcrit=200,
                burningRate=5,
                heatContent=21,
                planarDiffusivity=0.05,
                atmosphericDiffusivity=0.04,
                slopeContribution=1)        # Initialize fields and parameters
sim.Run(animStep=20)                                   # Perform the simulation
#sim.Show()                                  # Visualize the results
