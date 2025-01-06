import numpy as np
import random as r
import matplotlib.pyplot as plt

class Lattice:
    def __init__(self, latdims, initConfig=None,m=1,l=0, dMax0 = 1,warmCycles = None):
        self.latdims = latdims
        self.Ntot = np.prod(self.latdims)
        self.dim = len(self.latdims)
        self.lat = np.zeros(self.Ntot) if initConfig is None else np.array(initConfig)
        self.addressList = np.arange(self.Ntot)
        self.dMax = dMax0
        self.m = m
        self.l = l


        self.vec = np.ones(self.dim + 1)*self.Ntot

        for v in range(self.dim):
            self.vec[v+1] = self.vec[v]/latdims[v]


        if warmCycles is not None:
            self.metroCycles(warmCycles)
    
    def scrambleLattice(self):
        for n in range(self.Ntot):
            self.lat[n] = r.gauss(0,1)


    def show(self):
        showLat = np.reshape(self.lat,self.latdims)
        print(showLat)
    
    def shuffleList(self):
        
        indices = np.random.permutation(self.Ntot)
        self.addressList = self.addressList[indices]
    
    def expectation(self,func=None):
        
        # If func is None, use identity
        if func is None:
            func = lambda x: x

        M = 0
        for n in range(self.Ntot):
            M += func(self.lat[n])

        M = M/self.Ntot
        return M

    def shift(self,site,dim,jump):
        shiftedSite = int(self.vec[dim]* (site //self.vec[dim]) + (site + jump * self.vec[dim+1] + self.Ntot) % (self.vec[dim]))
        return shiftedSite

    def getNeighbours(self,site):
        neighbs = np.zeros(2*self.dim)
        for i in range(2*self.dim):
            d = int(np.ceil((i+1)/2))-1

            jump = (-1)**i
            neighbs[i] = self.shift(site,d,jump)
        return neighbs



    def findAction(self):
        S = 0
        dS = 0
        
        for n in range(self.Ntot):
            dS = 0
            neighbTotal = 0
            for d in range(self.dim):
                neighbTotal += self.lat[self.shift(n,d,1)]
            
            dS += (self.dim + (self.m**2)/2)*self.lat[n]**2 - self.lat[n]*neighbTotal + (self.l / 24) * self.lat[n]**4 
            
            S += dS
        return S


    def actionChange(self, address,d):


        neighbours = self.getNeighbours(address)

        

        neighbSum = 0
        for j in range(2*self.dim):
            n = int(neighbours[j])
            neighbSum += self.lat[n]

        dS = d*( 2*self.lat[address]*(2+ self.m*self.m/2) - neighbSum  ) \
        + d*d*(2 + self.m*self.m/2 )


        #dSl = self.l*( \
        #+ d*np.power(self.lat[address],3)/6  \
        #+ d*d*np.power(self.lat[address],2)/4 \
        #+ d*d*d*self.lat[address]/6 \
        #+ d*d*d*d/24 )

        return dS #+dSl




    def metroUpdate(self,n):

        d = r.gauss(0,self.dMax)

        

        dS = self.actionChange(n,d)



        p = min(1,np.exp(-dS))

        roll = r.uniform(0,1)

        if roll <= p:
            self.lat[n] += d
    
    def metroSweep(self):
        #self.shuffleList()


        for i in range(self.Ntot):
            n = self.addressList[i]
            self.metroUpdate(n)

    def metroCycles(self,cycles):
        for c in range(cycles):
            self.metroSweep()

    def twoPointCorr(self):
    
        C = 0


        
        CArray = np.zeros(self.Ntot**2)
        number = 0

        for n1 in range(self.Ntot):
            for n2 in range(self.Ntot):
                CArray[number] = self.lat[n1]*self.lat[n2]
                number += 1

        C = np.mean(CArray)
        CError = np.std(CArray)/(self.Ntot-1)
        return C, CError
    
    def twoPointTimeCorr(self, configNumber, interconfigCycles):

        timesteps, spacesteps = self.latdims
        
        GCMatrix = np.zeros((timesteps+1,configNumber))

        for i in range(configNumber):
            self.metroCycles(interconfigCycles)
            
            for tau2 in range(timesteps+1):
                tau = tau2 % timesteps
                corr = 0.0
                
                for t2 in range(timesteps+1):
                    t = t2%timesteps
                    for x in range(spacesteps):
                        for y in range(spacesteps):
                            n1 = t * spacesteps + x
                            n2 = ((t + tau) % timesteps) * spacesteps + y
                            corr += self.lat[n1] * self.lat[n2]
                

                corr /= (spacesteps * spacesteps * timesteps)
                GCMatrix[tau2,i] += corr

            print("Done config ",i, "/",configNumber)

        # Average over configurations
        GCArray = np.mean(GCMatrix, axis=1)
        GCErrors = np.std(GCMatrix, axis=1)/np.sqrt(configNumber-1)

        return GCArray, GCErrors