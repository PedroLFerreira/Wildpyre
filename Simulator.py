import numpy as np
import time


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
            self.T = np.zeros((nt,nx,ny))
        else:
            self.T = T
        if F is None:
            self.F = np.full((nt,nx,ny),500.0)
        else:
            self.F = F
        if A is None:
            self.A = np.zeros((nx,ny))
        else:
            self.A = A
        if H is None:
            self.H = np.zeros((nt,nx,ny))
        else:
            self.H = H
        
        self.W = np.zeros(shape=(nt,nx,ny,2))
        self.W[:,:,:,0] = W[0]
        self.W[:,:,:,1] = W[1]
        self.Tcrit = Tcrit
        self.burningRate = burningRate
        self.heatContent = heatContent
        self.planarDiffusivity = planarDiffusivity
        self.atmosphericDiffusivity = atmosphericDiffusivity
        self.slopeContribution = slopeContribution

    def Run(self):
        begin = time.time()
        for t in range(self.nt-1):
            print("   {:.4}%".format(t/self.nt*100),end='\r') # Print progress to console
            self.T[t+1,1:-1,1:-1] = (self.T[t,1:-1,1:-1] + self.dt*(self.planarDiffusivity*(self.T[t,2:,1:-1]*(1-self.W[t,1:-1,1:-1,0]-(self.A[2:,1:-1]-self.A[1:-1,1:-1])) - 2*self.T[t,1:-1,1:-1] + self.T[t,:-2,1:-1]*(1+self.W[t,1:-1,1:-1,0]-(self.A[:-2,1:-1]-self.A[1:-1,1:-1]))) 
                                                                  + self.planarDiffusivity*(self.T[t,1:-1,2:]*(1-self.W[t,1:-1,1:-1,1]-(self.A[1:-1,2:]-self.A[1:-1,1:-1])) - 2*self.T[t,1:-1,1:-1] + self.T[t,1:-1,:-2]*(1+self.W[t,1:-1,1:-1,1]-(self.A[1:-1,:-2]-self.A[1:-1,1:-1])))
                                                                  - self.atmosphericDiffusivity*self.T[t,1:-1,1:-1] 
                                                                  + self.H[t,1:-1,1:-1])) # Heavily modified heat equation solved here
            Hot = self.T[t+1] > self.Tcrit       # Check where T is above Tcrit and store it in the boolean vector Hot
            self.F[t+1] = self.F[t]              # Copy the last F field state

            self.F[t+1][Hot] -= self.dt * self.F[t][Hot] *  self.burningRate * (self.T[t+1][Hot] - self.Tcrit)/(self.T[t+1][Hot] + self.Tcrit)     # Burn F if Hot
            self.F[t+1][Hot] = np.maximum(self.F[t+1][Hot], 0)      # Make sure F is always non-negative

            self.H[t+1][Hot] = self.dt * self.F[t][Hot] * self.burningRate * self.heatContent * (self.T[t+1][Hot] - self.Tcrit)/(self.T[t+1][Hot] + self.Tcrit) # Increase value in the H field if Hot
            self.H[t+1][np.logical_not(Hot)] = self.H[t][np.logical_not(Hot)]   # Carry on if not Hot
        print('Simulation took {} seconds.'.format(time.time() - begin))

    def Show(self):
        pass