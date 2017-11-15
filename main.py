from Simulator import Simulator
import numpy as np
from scipy import misc 

A = misc.imread('Terrain.png').T
A = A/np.max(A)
nx=1081;ny=1081;nt=100;       # Fundamental simulation parameters
T = np.zeros((nt,nx,ny))    # Initialize T field
T[0,int(nx/2):int(nx/2+4),int(ny/2):int(ny/2+4)] = 300      # Increase temperature in the middle of the grid


sim = Simulator(nx,ny,nt,T=T,A=A)   # Initialize fields and parameters
sim.Run()                       # Perform the simulation
sim.Show()                      # Visualize the results
