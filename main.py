from Simulator import Simulator
import numpy as np
from scipy import misc 

A = misc.imread('Terrain.png')[200:400,200:400].T
A = A/np.max(A)
nx=A.shape[0];ny=A.shape[0];nt=500;                         # Fundamental simulation parameters
T = np.zeros((nt,nx,ny))                                    # Initialize T field
T[0,int(nx/2):int(nx/2+2),int(ny/2):int(ny/2+3)] = 1000      # Increase temperature in the middle of the grid
F = np.random.normal(loc=500, scale=120, size=(nt,nx,ny))

sim = Simulator(nx,ny,nt,T=T,A=A,F=F,
                Tcrit=200,
                burningRate=5,
                heatContent=22,
                planarDiffusivity=1.2,
                atmosphericDiffusivity=.56,
                slopeContribution=1)        # Initialize fields and parameters
sim.Run()                                   # Perform the simulation
sim.Show()                                  # Visualize the results
