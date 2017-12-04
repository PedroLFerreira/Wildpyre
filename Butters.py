from Simulator import Simulator
import numpy as np
from scipy import misc 
import matplotlib.pyplot as plt

Terrain = misc.imread('Terrain.png').T
plt.imshow(Terrain,cmap='gray',origin='lower')
plt.show()
A = misc.imread('Terrain.png').T[850:1000,150:300]
plt.imshow(A,cmap='gray',origin='lower')
plt.colorbar()
plt.show()
nx=A.shape[0];ny=A.shape[0];nt=20000;                    # Fundamental simulation parameters
T = np.zeros((nx,ny))                                    # Initialize T field
T[nx//2, ny//2:ny//2+2] = 1000
F = np.random.normal(loc=500, scale=0, size=(nx,ny))
F = np.maximum(F, 0)

sim = Simulator(nx,ny,nt,T=T,A=A,F=F,dt=.01,
                Tcrit=100,
                burningRate=0.025,
                heatContent=90,
                planarDiffusivity=0.005,
                atmosphericDiffusivity=0.001,
                slopeContribution=0.001)         # Initialize fields and parameters
#sim.Run(animStep=20, Tclim=(0,800))             # Perform the simulation
#sim.Show()                                      # Visualize the results

sim.CreateGIF(skip=400, maxIterations = 1500,
                        Tclim=(0,500),
                        Hclim=(0,7),
                        TOnly=True,
                        name='butters.mp4',
                        checkpoints=4) # Creates video file