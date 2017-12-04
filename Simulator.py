import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from pylab import *
import copy
from datetime import timedelta

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
        newH[np.logical_not(Hot)] = 0   # Carry on if not Hot

        self.oldT[:,:], self.T[:,:] = self.T, newT
        self.oldF[:,:], self.F[:,:] = self.F, newF
        self.oldH[:,:], self.H[:,:] = self.H, newH

    def Run(self, animStep=100,verbose=1,
                               Tclim=None,
                               Hclim=None,
                               Fclim=None):
        if animStep != 0:
            self._BeginAnimation(Tclim=Tclim,
                                 Hclim=Hclim,
                                 Fclim=Fclim)
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
        if verbose:
            print('Simulation took {} seconds.'.format(time.time() - begin))
        if animStep != 0:
            plt.close(self.fig)

    def _BeginAnimation(self, Tclim=None,
                              Hclim=None,
                              Fclim=None):
        if Tclim==None:
            Tclim = (0, self.Tcrit*2)
        if Hclim==None:
            Hclim = (0, 200*self.burningRate*self.heatContent*self.dt)
        if Fclim==None:
            Fclim=(0, 1000)

        self.fig = plt.figure(figsize=(15, 4))
        get_current_fig_manager().window.wm_geometry("+0+0")

        ax1 = self.fig.add_subplot(131)
        ax1.set_title('Temperature')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.imshow(self.A.T, cmap='Greys_r', origin='lower')
        cmap = cm.get_cmap('viridis')
        Tcmap = cmap(np.arange(cmap.N))
        Tcmap[:,-1] = np.append(np.linspace(0,1,cmap.N//4),np.ones(cmap.N-cmap.N//4))
        Tcmap = ListedColormap(Tcmap)
        self.TeImg = plt.imshow(self.T.T, cmap=Tcmap, origin='lower')
        plt.colorbar(self.TeImg)
        ax1.set_autoscale_on(True)
        plt.clim(Tclim)

        ax2 = self.fig.add_subplot(132)
        ax2.set_title('Fuel')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.imshow(self.A.T, cmap='Greys_r', origin='lower')
        cmap = cm.get_cmap('copper')
        Fcmap = cmap(np.arange(cmap.N))
        Fcmap[:,-1] = np.linspace(0,1,cmap.N)
        Fcmap = ListedColormap(Fcmap)
        self.FuImg = plt.imshow(self.F.T, cmap=Fcmap, origin='lower')
        plt.colorbar(self.FuImg)
        plt.clim(Fclim)
        self.fig.show()

        ax3 = self.fig.add_subplot(133)
        ax3.set_title('Heat')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.imshow(self.A.T, cmap='Greys_r', origin='lower')
        cmap = cm.get_cmap('inferno')
        Hcmap = cmap(np.arange(cmap.N))
        Hcmap[:,-1] = np.append(np.linspace(0,1,cmap.N//4),np.ones(cmap.N-cmap.N//4))
        Hcmap = ListedColormap(Hcmap)
        self.HImg = plt.imshow(self.H.T,cmap=Hcmap,origin='lower')
        plt.colorbar(self.HImg)
        plt.clim(Hclim)
        self.fig.show()

    def _UpdateAnimation(self, t=''):
        print(t,end='\r')
        self.TeImg.set_data(self.T.T)
        self.FuImg.set_data(self.F.T)
        self.HImg.set_data(self.H.T)
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

    def CreateGIF(self, skip=20, maxIterations=100,Tclim=None,
                                                   Hclim=None,
                                                   Fclim=None,
                                                   TOnly=False,
                                                   FOnly=False,
                                                   TandF=False,
                                                   checkpoints=1,
                                                   blackBG=False, #background color of colorbar as black
                                                   name='changethis.mp4'):
        if Tclim==None:
            Tclim = (0, self.Tcrit*2)
        if Hclim==None:
            Hclim = (0, 200*self.burningRate*self.heatContent*self.dt)
        if Fclim==None:
            Fclim=(0, 1000)
        fig = plt.figure()
        
        if TOnly:
            ax1 = fig.add_subplot(111)
            ax1.set_title('Temperature')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.imshow(self.A, cmap='Greys_r', origin='lower')  
            cmap = cm.get_cmap('viridis')
            Tcmap = cmap(np.arange(cmap.N))
            Tcmap[:,-1] = np.append(np.linspace(0,1,cmap.N//4),np.ones(cmap.N-cmap.N//4))
            Tcmap = ListedColormap(Tcmap)
            self.TeImg = plt.imshow(self.T.T, cmap=Tcmap, origin='lower', interpolation='nearest')
            cbT = plt.colorbar(self.TeImg)
            ax1.set_autoscale_on(True)
            plt.clim(Tclim)
            if blackBG:
                cbT.patch.set_facecolor((0, 0, 0, 1.0))
        elif FOnly:
            ax1 = fig.add_subplot(111)
            ax1.set_title('Fuel')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.imshow(self.A, cmap='Greys_r', origin='lower')
            cmap = cm.get_cmap('copper')
            Fcmap = cmap(np.arange(cmap.N))
            Fcmap[:,-1] = np.linspace(0,1,cmap.N)
            Fcmap = ListedColormap(Fcmap)
            self.FuImg = plt.imshow(self.F.T, cmap=Fcmap, origin='lower')
            cbF = plt.colorbar(self.FuImg)
            plt.clim(Fclim)
            if blackBG:
                cbF.patch.set_facecolor((0, 0, 0, 1.0))
        elif TandF:
            fig = plt.figure(figsize=(5,8))
            ax1 = fig.add_subplot(211)
            ax1.set_title('Temperature')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.imshow(self.A, cmap='Greys_r', origin='lower')  
            cmap = cm.get_cmap('viridis')
            Tcmap = cmap(np.arange(cmap.N))
            Tcmap[:,-1] = np.append(np.linspace(0,1,cmap.N//4),np.ones(cmap.N-cmap.N//4))
            Tcmap = ListedColormap(Tcmap)
            self.TeImg = plt.imshow(self.T.T, cmap=Tcmap, origin='lower', interpolation='nearest')
            cbT = plt.colorbar(self.TeImg)
            ax1.set_autoscale_on(True)
            plt.clim(Tclim)
            if blackBG:
                cbT.patch.set_facecolor((0, 0, 0, 1.0))

            ax2 = fig.add_subplot(212)
            #plt.sca(ax2)
            ax2.set_title('Fuel')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.imshow(self.A, cmap='Greys_r', origin='lower')
            cmap = cm.get_cmap('copper')
            Fcmap = cmap(np.arange(cmap.N))
            Fcmap[:,-1] = np.linspace(0,1,cmap.N)
            Fcmap = ListedColormap(Fcmap)
            self.FuImg = plt.imshow(self.F.T, cmap=Fcmap, origin='lower')
            cbF = plt.colorbar(self.FuImg)
            plt.clim(Fclim)
            if blackBG:
                cbF.patch.set_facecolor((0, 0, 0, 1.0))
        else:
            fig = plt.figure(figsize=(3,7))
            ax1 = fig.add_subplot(311)
            ax1.set_title('Temperature')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.imshow(self.A, cmap='Greys_r', origin='lower')  
            cmap = cm.get_cmap('viridis')
            Tcmap = cmap(np.arange(cmap.N))
            Tcmap[:,-1] = np.append(np.linspace(0,1,cmap.N//4),np.ones(cmap.N-cmap.N//4))
            Tcmap = ListedColormap(Tcmap)
            self.TeImg = plt.imshow(self.T.T, cmap=Tcmap, origin='lower', interpolation='nearest')
            cbT = plt.colorbar(self.TeImg)
            ax1.set_autoscale_on(True)
            plt.clim(Tclim)
            if blackBG:
                cbT = cbT.patch.set_facecolor((0, 0, 0, 1.0))

            ax2 = fig.add_subplot(312)
            ax2.set_title('Fuel')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.imshow(self.A, cmap='Greys_r', origin='lower')
            cmap = cm.get_cmap('copper')
            Fcmap = cmap(np.arange(cmap.N))
            Fcmap[:,-1] = np.linspace(0,1,cmap.N)
            Fcmap = ListedColormap(Fcmap)
            self.FuImg = plt.imshow(self.F.T, cmap=Fcmap, origin='lower')
            cbF = plt.colorbar(self.FuImg)
            plt.clim(Fclim)
            if blackBG:
                cbF.patch.set_facecolor((0, 0, 0, 1.0))

            ax3 = fig.add_subplot(313)
            ax3.set_title('Heat')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.imshow(self.A, cmap='Greys_r', origin='lower')
            cmap = cm.get_cmap('inferno')
            Hcmap = cmap(np.arange(cmap.N))
            Hcmap[:,-1] = np.append(np.linspace(0,1,cmap.N//4),np.ones(cmap.N-cmap.N//4))
            Hcmap = ListedColormap(Hcmap)
            self.HImg = plt.imshow(self.H.T,cmap=Hcmap,origin='lower')
            cbH = plt.colorbar(self.HImg)
            plt.clim(Hclim)
            if blackBG:
                cbH.patch.set_facecolor((0, 0, 0, 1.0))

        plt.tight_layout()
        
        estimatedTime = 0
        initialTime = time.time()
        def update(i):
            estimatedTime = (time.time()-initialTime)*(maxIterations/(i+1e-5)-1)
            for t in range(skip):
                self._Step()
            
            if not FOnly:
                self.TeImg.set_data(self.T.T)
            if not TOnly:
                self.FuImg.set_data(self.F.T)
            if not FOnly and not TOnly and not TandF:
                self.HImg.set_data(self.H.T)
            print(' ETA: {}                     '.format(timedelta(seconds=estimatedTime)), end='\r')
            if checkpoints:
                if i % (maxIterations//checkpoints) == (maxIterations//checkpoints-1):
                    plt.savefig(name.split('.')[0] +'_fig' + str(i // (maxIterations//checkpoints)) + '.pdf')
            if TOnly:
                return self.TeImg,
            if FOnly:
                return self.FuImg,
            if TandF:
                return (self.TeImg, self.FuImg)
            return (self.TeImg, self.FuImg, self.HImg)

        anim = FuncAnimation(fig, update, frames=np.arange(0,maxIterations), interval=10, blit=True)
        anim.save(name, dpi=200, writer='ffmpeg')
