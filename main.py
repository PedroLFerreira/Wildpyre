from Simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

nx=100;ny=100;nt=500;       # Fundamental simulation parameters
T = np.zeros((nt,nx,ny))    # Initialize T field
T[0,int(nx/2):int(nx/2+4),int(ny/2):int(ny/2+4)] = 300      # Increase temperature in the middle of the grid

sim = Simulator(nx,ny,nt,T=T)   # Initialize fields and parameters
sim.Run()                       # Perform the simulation
sim.Show()                      # This does nothing yet but should replace all the cancer below

#=================VISUALIZATION SHOULD GO INSIDE THE CLASS BECAUSE IT LOOKS LIKE CANCER============================#
#plt.ion()
fig = plt.figure(figsize=(10, 10))
get_current_fig_manager().window.wm_geometry("+0+0")

ax1 = fig.add_subplot(221)
ax1.set_title('Temperature')
plt.xlabel('x')
plt.ylabel('y')
TeImg = plt.imshow(sim.T[0].T, cmap='viridis', origin='lower')
plt.colorbar(TeImg)
ax1.set_autoscale_on(True)
plt.clim(0, 500)

ax2 = fig.add_subplot(222)
plt.xlabel('x')
plt.ylabel('y')
ax2.set_title('Heat')
plt.xlabel('y')
plt.ylabel('x')
HeImg = plt.imshow(sim.H[0].T, cmap='inferno', origin='lower')
plt.colorbar(HeImg)
plt.clim(0, 500)
fig.show()

ax2 = fig.add_subplot(223)
ax2.set_title('Fuel')
plt.xlabel('y')
plt.ylabel('x')
FuImg = plt.imshow(sim.F[0].T, cmap='copper', origin='lower')
plt.colorbar(FuImg)
plt.clim(0, 1000)
fig.show()

ax4 = fig.add_subplot(224)
plt.xlabel('x')
plt.ylabel('y')
ax4.set_title('Altitude')
AlImg = plt.imshow(sim.A.T, cmap='viridis', origin='lower')
plt.colorbar(AlImg)
plt.clim(0, 1)
fig.show()

for t in range(0,nt,2):
    print(t,end='\r')
    TeImg.set_data(sim.T[t].T)
    HeImg.set_data(sim.H[t].T)
    FuImg.set_data(sim.F[t].T)
    fig.canvas.draw()
    #plt.pause(1e-2)
