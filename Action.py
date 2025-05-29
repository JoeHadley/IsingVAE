import numpy as np
import random as r

class Action:
    def __init__(self, m=1,l=0, dMax0 = 1):
        

        self.dMax = dMax0
        self.m = m
        self.l = l
 

    def findAction(self,Lattice,workingLattice):
        S = 0
        dS = 0
        
        for n in range(self.Ntot):
            dS = 0
            neighbTotal = 0
            for d in range(Lattice.dim):
                neighbTotal += workingLattice[Lattice.shift(n,d,1)]
            

            #dS += (self.dim + (self.m**2)/2)*self.lat[n]**2 - self.lat[n]*neighbTotal + (self.l / 24) * self.lat[n]**4 
            dS += (Lattice.dim + (self.m**2)/2)*workingLattice[n]**2 - workingLattice[n]*neighbTotal + (self.l / 24) * workingLattice[n]**4 
            
            S += dS
        return S

    def actionChange(self,Lattice, workingLattice, address,d):


        neighbours = Lattice.getNeighbours(address)

        

        neighbSum = 0
        for j in range(2*self.dim): # Not currently general for other lattice shapes
            n = int(neighbours[j])
            neighbSum += workingLattice[n]

        dS = d*( 2*workingLattice[address]*(2+ self.m*self.m/2) - neighbSum  ) \
        + d*d*(2 + self.m*self.m/2 )


        dSl = self.l*( \
        + d*np.power(self.lat[address],3)/6  \
        + d*d*np.power(self.lat[address],2)/4 \
        + d*d*d*self.lat[address]/6 \
        + d*d*d*d/24 )

        return dS +dSl
