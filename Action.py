import numpy as np
import random as r

class Action:
    def __init__(self, m=1.0,l=0):
        

        self.m = m
        self.l = l


 
    def findAction(self,simulation,overrideWorkingLattice=None):
        lattice = simulation.lattice
        workingLattice = simulation.workingLattice if overrideWorkingLattice is None else overrideWorkingLattice
        dim = simulation.lattice.dim
        S = 0
        
        
        for n in range(lattice.Ntot):
            
            phi = workingLattice[n]
            neighbTotal = 0

            # Try use both sides 
            for d in range(dim):
                neighbTotal += workingLattice[lattice.shift(n,d,1)]
                neighbTotal += workingLattice[lattice.shift(n,d,-1)]
            
            kinetic = lattice.dim * phi**2 - phi * neighbTotal /2  # Divide neighbTotal by 2

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
        neighbour_sum = 0
        for d in range(lattice.dim):
            neighbour_sum += workingLattice[lattice.shift(site, d, 1)]
            if not forwardOnly:
                neighbour_sum += workingLattice[lattice.shift(site, d,-1)]
        return neighbour_sum

    def actionChange(self, simulation, site,d):
        lattice = simulation.lattice
        workingLattice = simulation.workingLattice
        dim = lattice.dim

        neighbours = lattice.getNeighbours(site)

        

        neighbSum = 0
        for j in range(2*dim):
            n = int(neighbours[j])
            neighbSum += workingLattice[n]

        dS = d*( 2*workingLattice[site]*(dim+ self.m*self.m/2) - neighbSum  ) \
        + d*d*(dim + self.m*self.m/2 )


        dSl = self.l*( \
        + d*np.power(workingLattice[site],3)/6  \
        + d*d*np.power(workingLattice[site],2)/4 \
        + d*d*d*workingLattice[site]/6 \
        + d*d*d*d/24 )

        change_in_action = dS + dSl

        return change_in_action
    
    def actionChangeLong(self, simulation, site,d):
        lattice = simulation.lattice
        workingLattice = simulation.workingLattice


        #directly calculate the change in action for a given site and displacement
        OldAction = self.findAction(simulation)  # Current action value


        trial = workingLattice.copy()
        trial[site] += d
        NewAction = self.findAction(simulation,overrideWorkingLattice=trial )  # Action value after the change
        return NewAction - OldAction
    


