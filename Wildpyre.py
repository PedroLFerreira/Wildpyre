import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from scipy import misc
import matplotlib.animation as animation

#np.random.seed(42)

xmin = 0; xmax = 10; nx = 51; dx = 2/(nx - 1)
ymin = 0; ymax = 10; ny = 51; dy = 2/(ny - 1)
nt = 200; dt = .1

X, Y = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
Te = np.zeros((nt, nx, ny))         #Temperature
Fi = np.zeros((nt, nx, ny))         #Fire
Fu = np.zeros((nt, nx, ny))         #Fuel Mass
W = np.ones((nt, nx, ny, 2))          #Wind
#G = np.zeros((nt, nx, ny))         #Ground Types
#H = np.zeros((nt, nx, ny))         #Height


Tcrit = 100                     # Temperature at which fuel ignites
burningRate = .005             # How fast fuel burns
heatContent = 2                 # How much heat the fuel generates
planarDiffusivity = 20          # How fast T diffuses to the surrounding terrain
atmosphericDiffusivity = 1      # How fast T diffuses to the atmosphere
fireContribution = 10           # How much T increases due to the fire
maximumBurning = 100

Te[0,int(nx/2):int(nx/2+4),int(ny/2):int(ny/2+2)] = 650

W[:,:,:,0] = 0                  # Wind in positive-x direction
W[:,:,:,1] = .2                 # Wind in positive-y direction

Fu[0] = np.full((nx,ny),1000) - np.random.uniform(0, 400, size=(nx, ny))

for t in range(nt-1):
    print("   {:.4}%".format(t/nt*100),end='\r')
    for x in range(1,nx-1):
        for y in range(1,ny-1):
            windAngle = np.arctan2(W[t,x,y,1], W[t,x,y,0])
            windStrength = np.sqrt(W[t,x,y,1]**2 + W[t,x,y,0]**2)

            # Diffuse temperature through the plane and to the atmosphere and add the fire heat contribution
            Te[t+1,x,y] = (Te[t,x,y] + dt*(planarDiffusivity*dx*(Te[t,x+1,y]*(1-windStrength*np.cos(windAngle)) - 2*Te[t,x,y] + Te[t,x-1,y]*(1+windStrength*np.cos(windAngle)))
                                         + planarDiffusivity*dy*(Te[t,x,y+1]*(1-windStrength*np.sin(windAngle)) - 2*Te[t,x,y] + Te[t,x,y-1]*(1+windStrength*np.sin(windAngle)))
                                         - atmosphericDiffusivity*Te[t,x,y]
                                         + fireContribution*Fi[t,x,y]))

            if Te[t+1,x,y] > Tcrit: # Cell will ignite or continue burning
                # Fuel burns proportional to the amount that exists and delta T * burningRate
                Fu[t+1,x,y] = Fu[t,x,y] - dt * Fu[t,x,y]*max(Te[t+1,x,y] - Tcrit, maximumBurning)* burningRate
                Fu[t+1,x,y] = max(Fu[t+1,x,y], 0)
                # Heat proportional to the mass of burnt fuel
                Fi[t+1,x,y] = dt * Fu[t,x,y] * (Te[t+1,x,y] - Tcrit) * burningRate * heatContent    
            else:
                Fu[t+1,x,y] = Fu[t,x,y]
                Fi[t+1,x,y] = Fi[t,x,y]



#plt.ion()
fig = plt.figure(figsize=(20, 4))
ax1 = fig.add_subplot(131)
ax1.set_title('Temperature')
plt.xlabel('y')
plt.ylabel('x')
TeImg = plt.imshow(Te[0], cmap='viridis', origin='lower')
plt.colorbar(TeImg)
ax1.set_autoscale_on(True)
plt.clim(0, 1500)

ax2 = fig.add_subplot(132)
ax2.set_title('Heat')
FiImg = plt.imshow(Fi[0], cmap='inferno', origin='lower')
plt.colorbar(FiImg)
plt.clim(0, 500)
fig.show()

ax2 = fig.add_subplot(133)
ax2.set_title('Fuel')
FuImg = plt.imshow(Fu[0], cmap='copper', origin='lower')
plt.colorbar(FuImg)
plt.clim(0, 1000)
fig.show()

for t in range(0,nt,2):
    print(t,end='\r')
    TeImg.set_data(Te[t])
    FiImg.set_data(Fi[t])
    FuImg.set_data(Fu[t])
    fig.canvas.draw()
    plt.pause(1e-17)
