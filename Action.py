import numpy as np
import random as r

class Action:
    def __init__(self, m=1.0,l=0):
        

        self.m = m
        self.l = l


 
    def findAction(self,simulation,overrideWorkingLattice=None):
        lattice = simulation.lattice
        workingLattice = simulation.workingLattice if overrideWorkingLattice is None else overrideWorkingLattice
        beta = simulation.beta
        dim = simulation.lattice.dim
        S = 0
        
        
        for n in range(lattice.Ntot):
            
            phi = workingLattice[n]
            neighbTotal = 0

            # Try use both sides 
            for d in range(dim):
                neighbTotal += workingLattice[lattice.shift(n,d,1)]
                neighbTotal += workingLattice[lattice.shift(n,d,-1)]
            
            kinetic = lattice.dim * phi**2 - phi * neighbTotal

            mass_term = 0.5 * self.m**2 * phi**2
            quartic = (self.l / 24) * phi**4

            S += kinetic + mass_term + quartic
        return S


        


    def sumNeighbours(self, simulation, site,overrideLattice=None,forwardOnly=False):
        
        
        lattice = simulation.lattice

        if overrideLattice is not None:
            workingLattice = overrideLattice
        else:
            workingLattice = simulation.workingLattice

        # Sum the values of the neighbours of a given site
        neighbSum = 0
        for d in range(lattice.dim):
            neighbSum += workingLattice[lattice.shift(site, d, 1)]
            if not forwardOnly:
                neighbSum += workingLattice[lattice.shift(site, d,-1)]
        return neighbSum

    def actionChange(self, simulation, address,d):
        lattice = simulation.lattice
        workingLattice = simulation.workingLattice
        dim = lattice.dim

        neighbours = lattice.getNeighbours(address)

        

        neighbSum = 0
        for j in range(2*dim): # Not currently general for other lattice shapes
            n = int(neighbours[j])
            neighbSum += workingLattice[n]

        dS = d*( 2*workingLattice[address]*(2+ self.m*self.m/2) - neighbSum  ) \
        + d*d*(2 + self.m*self.m/2 )


        dSl = self.l*( \
        + d*np.power(workingLattice[address],3)/6  \
        + d*d*np.power(workingLattice[address],2)/4 \
        + d*d*d*workingLattice[address]/6 \
        + d*d*d*d/24 )

        return dS +dSl
    
    def actionChangeLong(self, simulation, address,d):
        lattice = simulation.lattice
        workingLattice = simulation.workingLattice


        #directly calculate the change in action for a given site and displacement
        OldAction = self.findAction(simulation)  # Current action value

        #Copy the working lattice to avoid modifying it directly
        trial = workingLattice.copy()
        trial[address] += d
        NewAction = self.findAction(simulation,overrideWorkingLattice=trial )  # Action value after the change
        return NewAction - OldAction
    


