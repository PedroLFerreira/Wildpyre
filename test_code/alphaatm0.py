from Simulator import Simulator
import numpy as np
from scipy import misc 
import matplotlib.pyplot as plt

A = np.zeros((50,50))
nx=A.shape[0];ny=A.shape[0];nt=100000;                         # Fundamental simulation parameters
T = np.zeros((nx,ny))                                    # Initialize T field
#T[int(nx/2):int(nx/2+2),int(ny/2):int(ny/2+3)] = 1000      # Increase temperature in the middle of the grid
T[nx//2, ny//2] = 1000
F = np.random.normal(loc=500, scale=0, size=(nx,ny))
F = np.maximum(F, 0)
W = (0,0)

sim = Simulator(nx,ny,nt,T=T.copy(),A=A.copy(),F=F.copy(),dt=0.01, W=W,
                Tcrit=100,
                burningRate=0.025,
                heatContent=170,
                planarDiffusivity=0.01,
                atmosphericDiffusivity=0,
                slopeContribution=1)        # Initialize fields and parameters
#sim.Run(animStep=500, Tclim=(500,0))                                   # Perform the simulation
sim.CreateGIF(skip=100, maxIterations=1000, name='alphaatm0.mp4', Tclim=(500,0))