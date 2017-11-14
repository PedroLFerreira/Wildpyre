import numpy as np
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use("wx")
import matplotlib.pyplot as plt 
from scipy import misc
import matplotlib.animation as animation
import time
from pylab import *

#np.random.seed(42)


nx = 101 
ny = 101 
nt = 500
dt = .1

#X, Y = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
Te = np.zeros((nt, nx, ny))         #Temperature
He = np.zeros((nt, nx, ny))         #Heat
Fu = np.zeros((nt, nx, ny))         #Fuel Mass
W = np.ones((nt, nx, ny, 2))        #Wind

Tcrit = 100                     # Temperature at which fuel ignites
burningRate = 2             # How fast fuel burns
heatContent = 10               # How much heat the fuel generates
planarDiffusivity = 2           # How fast T diffuses to the surrounding terrain
atmosphericDiffusivity = .1      # How fast T diffuses to the atmosphere
#fireContribution = 10           # How much T increases due to the fire
slopeContribution = 1           # How much the terrain slope influences T planar diffusion

Te[0,int(nx/2):int(nx/2+4),int(ny/2):int(ny/2+4)] = 300

W[:,:,:,0] = 0                  # Wind in positive-x direction
W[:,:,:,1] = 0                  # Wind in positive-y direction
H = np.array([[0.1*np.sin(i/10+j/5)+.5-np.random.uniform(0,.1) for i in range(nx)] for j in range(ny)])

Fu[0] = np.full((nx,ny), 1000) - np.random.uniform(0, 600, size=(nx, ny))
Fu[0] = np.random.normal(500, 100, size=(nx,ny))

begin = time.time()

for t in range(nt-1):
    print("   {:.4}%".format(t/nt*100),end='\r')
    Te[t+1,1:-1,1:-1] = (Te[t,1:-1,1:-1] + dt*(planarDiffusivity*(Te[t,2:,1:-1]*(1-W[t,1:-1,1:-1,0]-(H[2:,1:-1]-H[1:-1,1:-1])) - 2*Te[t,1:-1,1:-1] + Te[t,:-2,1:-1]*(1+W[t,1:-1,1:-1,0]-(H[:-2,1:-1]-H[1:-1,1:-1])))
                                         + planarDiffusivity*(Te[t,1:-1,2:]*(1-W[t,1:-1,1:-1,1]-(H[1:-1,2:]-H[1:-1,1:-1])) - 2*Te[t,1:-1,1:-1] + Te[t,1:-1,:-2]*(1+W[t,1:-1,1:-1,1]-(H[1:-1,:-2]-H[1:-1,1:-1])))
                                         - atmosphericDiffusivity*Te[t,1:-1,1:-1]
                                         + He[t,1:-1,1:-1]))
    
    Hot = Te[t+1] > Tcrit   # Check where T is above Tcrit and store it in the boolean vector Hot
    Fu[t+1] = Fu[t]         # Copy the last Fu field state

    Fu[t+1][Hot] -= dt * Fu[t][Hot] *  burningRate * (Te[t+1][Hot] - Tcrit)/(Te[t+1][Hot] + Tcrit)     # Burn Fu if Hot
    Fu[t+1][Hot] = np.maximum(Fu[t+1][Hot], 0)                                # Make sure Fu is always non-negative

    He[t+1][Hot] = dt * Fu[t][Hot] * burningRate * heatContent * (Te[t+1][Hot] - Tcrit)/(Te[t+1][Hot] + Tcrit)
    He[t+1][np.logical_not(Hot)] = He[t][np.logical_not(Hot)]

print('Simulation took {} seconds.'.format(time.time() - begin))

#plt.ion()
fig = plt.figure(figsize=(10, 10))
get_current_fig_manager().window.wm_geometry("+0+0")

ax1 = fig.add_subplot(221)
ax1.set_title('Temperature')
plt.xlabel('x')
plt.ylabel('y')
TeImg = plt.imshow(Te[0].T, cmap='viridis', origin='lower')
plt.colorbar(TeImg)
ax1.set_autoscale_on(True)
#plt.clim(0, 300)

ax2 = fig.add_subplot(222)
plt.xlabel('x')
plt.ylabel('y')
ax2.set_title('Heat')
plt.xlabel('y')
plt.ylabel('x')
HeImg = plt.imshow(He[0].T, cmap='inferno', origin='lower')
plt.colorbar(HeImg)
plt.clim(0, 500)
fig.show()

ax2 = fig.add_subplot(223)
ax2.set_title('Fuel')
plt.xlabel('y')
plt.ylabel('x')
FuImg = plt.imshow(Fu[0].T, cmap='copper', origin='lower')
plt.colorbar(FuImg)
plt.clim(0, 1000)
fig.show()

ax4 = fig.add_subplot(224)
plt.xlabel('x')
plt.ylabel('y')
ax4.set_title('Altitude')
AlImg = plt.imshow(H.T, cmap='viridis', origin='lower')
plt.colorbar(AlImg)
plt.clim(0, 1)
fig.show()

for t in range(0,nt,2):
    print(t,end='\r')
    TeImg.set_data(Te[t].T)
    HeImg.set_data(He[t].T)
    FuImg.set_data(Fu[t].T)
    fig.canvas.draw()
    #plt.pause(1e-2)
