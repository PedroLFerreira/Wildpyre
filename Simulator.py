import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pylab import *
class Simulator:
    def __init__ (self, nx, ny, nt, dt=0.1, T=None,
                                            F=None,
                                            H=None,
                                            A=None,
                                            W=(0,0),
                                            Tcrit=100,
                                            burningRate=2,
                                            heatContent=10,
                                            planarDiffusivity=2,
                                            atmosphericDiffusivity=0.1,
                                            slopeContribution=1):
        self.nx = nx
        self.ny = ny 
        self.nt = nt
        self.dt = dt
        if T is None:
            self.T = np.zeros((nx,ny))
        else:
            self.T = T
        if F is None:
            self.F = np.full((nx,ny),500.0)
        else:
            self.F = F
        if A is None:
            self.A = np.zeros((nx,ny))
        else:
            self.A = A
        if H is None:
            self.H = np.zeros((nx,ny))
        else:
            self.H = H
        
        self.initT = self.T.copy()
        self.initF = self.F.copy()
        self.initH = self.H.copy()

        self.oldF = np.zeros_like(self.F)
        self.oldT = np.zeros_like(self.T)
        self.oldH = np.zeros_like(self.H)

        self.W = np.zeros(shape=(nx,ny,2))
        self.W[:,:,0] = W[0]
        self.W[:,:,1] = W[1]
        self.Tcrit = Tcrit
        self.burningRate = burningRate
        self.heatContent = heatContent
        self.planarDiffusivity = planarDiffusivity
        self.atmosphericDiffusivity = atmosphericDiffusivity
        self.slopeContribution = slopeContribution

    def _Step(self):
        newT = np.zeros_like(self.T)
        newF = np.zeros_like(self.F)
        newH = np.zeros_like(self.H)
        newT[1:-1,1:-1] = (self.T[1:-1,1:-1] + self.dt*(self.planarDiffusivity*(self.T[2:,1:-1]*(1-self.W[1:-1,1:-1,0]-self.slopeContribution*(self.A[2:,1:-1]-self.A[1:-1,1:-1])) - 2*self.T[1:-1,1:-1] + self.T[:-2,1:-1]*(1+self.W[1:-1,1:-1,0]-self.slopeContribution*(self.A[:-2,1:-1]-self.A[1:-1,1:-1]))) 
                                                      + self.planarDiffusivity*(self.T[1:-1,2:]*(1-self.W[1:-1,1:-1,1]-self.slopeContribution*(self.A[1:-1,2:]-self.A[1:-1,1:-1])) - 2*self.T[1:-1,1:-1] + self.T[1:-1,:-2]*(1+self.W[1:-1,1:-1,1]-self.slopeContribution*(self.A[1:-1,:-2]-self.A[1:-1,1:-1])))
                                                      - self.atmosphericDiffusivity*self.T[1:-1,1:-1] 
                                                      + self.H[1:-1,1:-1])) # Heavily modified heat equation solved here
        Hot = self.T > self.Tcrit       # Check where T is above Tcrit and store it in the boolean vector Hot
        newF[:,:] = self.F              # Copy the last F field state

        deltaF = self.dt * self.F[Hot] * self.burningRate * (self.T[Hot] - self.Tcrit) / self.T[Hot]

        newF[Hot] -= deltaF     # Burn F if Hot
        newF[Hot] = np.maximum(newF[Hot], 0)      # Make sure F is always non-negative

        newH[Hot] = deltaF * self.heatContent # Increase value in the H field if Hot
        newH[np.logical_not(Hot)] = self.H[np.logical_not(Hot)]   # Carry on if not Hot

        self.oldT[:,:], self.T[:,:] = self.T, newT
        self.oldF[:,:], self.F[:,:] = self.F, newF
        self.oldH[:,:], self.H[:,:] = self.H, newH

    def Run(self, animStep=100):
        if animStep != 0:
            self._BeginAnimation()
        self.burning = []

        begin = time.time()
        for t in range(self.nt-1):
            self._Step()

            if animStep != 0 and t%animStep == 0:
                self._UpdateAnimation(t)
            
            self.burning.append(np.sum(self.oldF) - np.sum(self.F))

            if self.burning[-1] == 0 and t != 0:
                break
        
        self.tFinal = t
        print('Simulation took {} seconds.'.format(time.time() - begin))
        if animStep != 0:
            plt.close(self.fig)

    def _BeginAnimation(self):
        self.fig = plt.figure(figsize=(12, 12))
        #get_current_fig_manager().window.wm_geometry("+0+0")

        ax1 = self.fig.add_subplot(221)
        ax1.set_title('Temperature')
        plt.xlabel('x')
        plt.ylabel('y')
        self.TeImg = plt.imshow(self.T.T, cmap='viridis', origin='lower')
        plt.colorbar(self.TeImg)
        ax1.set_autoscale_on(True)
        plt.clim(0, self.Tcrit)

        ax2 = self.fig.add_subplot(222)
        plt.xlabel('x')
        plt.ylabel('y')
        ax2.set_title('Heat')
        plt.xlabel('y')
        plt.ylabel('x')
        self.HeImg = plt.imshow(self.H.T, cmap='inferno', origin='lower')
        plt.colorbar(self.HeImg)
        plt.clim(0, 10*self.dt)
        self.fig.show()

        ax2 = self.fig.add_subplot(223)
        ax2.set_title('Fuel')
        plt.xlabel('y')
        plt.ylabel('x')
        self.FuImg = plt.imshow(self.F.T, cmap='copper', origin='lower')
        plt.colorbar(self.FuImg)
        plt.clim(0, 1000)
        self.fig.show()

        ax4 = self.fig.add_subplot(224)
        plt.xlabel('x')
        plt.ylabel('y')
        ax4.set_title('Altitude')
        AlImg = plt.imshow(self.A.T, cmap='viridis', origin='lower')
        plt.colorbar(AlImg)
        self.fig.show()
    
    def _UpdateAnimation(self, t=''):
        print(t,end='\r')
        self.TeImg.set_data(self.T.T)
        self.HeImg.set_data(self.H.T)
        self.FuImg.set_data(self.F.T)
        self.fig.tight_layout
        self.fig.canvas.draw()
        plt.pause(1e-20)

    def Metrics(self):
        metrics = {}
        metrics['totalBurnt'] = (np.sum(self.initF) - np.sum(self.F)) / np.sum(self.initF) 
        metrics['burntStep'] = np.array(self.burning)
        metrics['burning'] = (self.burning[-1] != 0)
        metrics['elapsedTime'] = self.tFinal
        
        if metrics['burning']:
            fitSize = len(metrics['burntStep'])
            fitRange = fitSize // 2
            m, b = np.polyfit(np.arange(fitSize-fitRange, fitSize), metrics['burntStep'][-fitRange:], deg=1)
            metrics['burnRate'] = (m, b)
        else:
            metrics['burnRate'] = None

        return metrics

    # def Run(self):
    #     begin = time.time()
    #     for t in range(self.nt-1):
    #         print("   {:.4}%".format(t/self.nt*100),end='\r') # Print progress to console
    #         self.T[t+1,1:-1,1:-1] = (self.T[t,1:-1,1:-1] + self.dt*(self.planarDiffusivity*(self.T[t,2:,1:-1]*(1-self.W[t,1:-1,1:-1,0]-self.slopeContribution*(self.A[2:,1:-1]-self.A[1:-1,1:-1])) - 2*self.T[t,1:-1,1:-1] + self.T[t,:-2,1:-1]*(1+self.W[t,1:-1,1:-1,0]-self.slopeContribution*(self.A[:-2,1:-1]-self.A[1:-1,1:-1]))) 
    #                                                               + self.planarDiffusivity*(self.T[t,1:-1,2:]*(1-self.W[t,1:-1,1:-1,1]-self.slopeContribution*(self.A[1:-1,2:]-self.A[1:-1,1:-1])) - 2*self.T[t,1:-1,1:-1] + self.T[t,1:-1,:-2]*(1+self.W[t,1:-1,1:-1,1]-self.slopeContribution*(self.A[1:-1,:-2]-self.A[1:-1,1:-1])))
    #                                                               - self.atmosphericDiffusivity*self.T[t,1:-1,1:-1] 
    #                                                               + self.H[t,1:-1,1:-1])) # Heavily modified heat equation solved here
    #         Hot = self.T[t+1] > self.Tcrit       # Check where T is above Tcrit and store it in the boolean vector Hot
    #         self.F[t+1] = self.F[t]              # Copy the last F field state

    #         self.F[t+1][Hot] -= self.dt * self.F[t][Hot] *  self.burningRate * (self.T[t+1][Hot] - self.Tcrit)/(self.T[t+1][Hot] + self.Tcrit)     # Burn F if Hot
    #         self.F[t+1][Hot] = np.maximum(self.F[t+1][Hot], 0)      # Make sure F is always non-negative

    #         self.H[t+1][Hot] = self.dt * self.F[t][Hot] * self.burningRate * self.heatContent * (self.T[t+1][Hot] - self.Tcrit)/(self.T[t+1][Hot] + self.Tcrit) # Increase value in the H field if Hot
    #         self.H[t+1][np.logical_not(Hot)] = self.H[t][np.logical_not(Hot)]   # Carry on if not Hot
    #     print('Simulation took {} seconds.'.format(time.time() - begin))

    # def Show(self):
    #     #plt.ion()
    #     fig = plt.figure(figsize=(12, 12))
    #     get_current_fig_manager().window.wm_geometry("+0+0")

    #     ax1 = fig.add_subplot(221)
    #     ax1.set_title('Temperature')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     TeImg = plt.imshow(self.T[0].T, cmap='viridis', origin='lower')
    #     plt.colorbar(TeImg)
    #     ax1.set_autoscale_on(True)
    #     plt.clim(0, 500)

    #     ax2 = fig.add_subplot(222)
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     ax2.set_title('Heat')
    #     plt.xlabel('y')
    #     plt.ylabel('x')
    #     HeImg = plt.imshow(self.H[0].T, cmap='inferno', origin='lower')
    #     plt.colorbar(HeImg)
    #     plt.clim(0, 500)
    #     fig.show()

    #     ax2 = fig.add_subplot(223)
    #     ax2.set_title('Fuel')
    #     plt.xlabel('y')
    #     plt.ylabel('x')
    #     FuImg = plt.imshow(self.F[0].T, cmap='copper', origin='lower')
    #     plt.colorbar(FuImg)
    #     plt.clim(0, 1000)
    #     fig.show()

    #     ax4 = fig.add_subplot(224)
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     ax4.set_title('Altitude')
    #     AlImg = plt.imshow(self.A.T, cmap='viridis', origin='lower')
    #     plt.colorbar(AlImg)
    #     fig.show()

    #     for t in range(0,self.nt,2):
    #         print(t,end='\r')
    #         TeImg.set_data(self.T[t].T)
    #         HeImg.set_data(self.H[t].T)
    #         FuImg.set_data(self.F[t].T)
    #         fig.tight_layout
    #         fig.canvas.draw()
    #         plt.pause(1e-20)
