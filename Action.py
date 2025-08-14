import numpy as np
import random as r

class Action:
    def __init__(self, m=1,l=0, dMax0 = 1):
        

        self.dMax = dMax0
        self.m = m
        self.l = l


        self.observables = {
            "phi4": lambda lattice, workingLattice: self.expectation(lattice, workingLattice, func=lambda x: x**4),
            "phi2": lambda lattice, workingLattice: self.expectation(lattice, workingLattice, func=lambda x: x**2),
            "phiBar": lambda lattice, workingLattice: self.expectation(lattice, workingLattice),
            "action": self.findAction,
            "empty": lambda lattice, workingLattice: 0,
        }
 
    def printParams(self):
        print(f"Action parameters: m = {self.m}, l = {self.l}, dMax = {self.dMax}")

    def findAction(self,lattice,workingLattice):
        S = 0
        
        
        for n in range(lattice.Ntot):
            
            phi = workingLattice[n]
            neighbTotal = 0
            for d in range(lattice.dim):
                neighbTotal += workingLattice[lattice.shift(n,d,1)]
            
            #kinetic = lattice.dim * phi**2 - phi * neighbTotal
            kinetic = 2*lattice.dim * phi**2 - 2 * phi * neighbTotal

            mass_term = 0.5 * self.m**2 * phi**2
            quartic = (self.l / 24) * phi**4

            #dS += (self.dim + (self.m**2)/2)*self.lat[n]**2 - self.lat[n]*neighbTotal + (self.l / 24) * self.lat[n]**4 
            #dS += (lattice.dim + (self.m**2)/2)*workingLattice[n]**2 - workingLattice[n]*neighbTotal + (self.l / 24) * workingLattice[n]**4 
            
            S += kinetic - mass_term + quartic
        return S

    def actionChange2(self,lattice, workingLattice, address,d):
        #directly calculate the change in action for a given site and displacement
        OldAction = self.findAction(lattice, workingLattice)
        #Copy the working lattice to avoid modifying it directly
        workingLattice2 = workingLattice.copy()
        workingLattice2[address] += d
        NewAction = self.findAction(lattice, workingLattice2)
        return NewAction - OldAction

    def sumNeighbours(self, simulation, site):
        lattice = simulation.lattice
        workingLattice = simulation.workingLattice

        # Sum the values of the neighbours of a given site
        neighbSum = 0
        for d in range(2*lattice.dim):
            neighbSum += workingLattice[lattice.shift(site, d, 1)]
            neighbSum += workingLattice[lattice.shift(site, d, -1)]
        return neighbSum

    def actionChange(self,lattice, workingLattice, address,d):

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
    
    def expectation(self, lattice, workingLattice, func=None):
        # If func is None, use identity
        if func is None:
            func = lambda x: x

        M = 0
        for n in range(lattice.Ntot):
            M += func(workingLattice[n])

        return M / lattice.Ntot
    
    def computeObservable(self, name, lattice, workingLattice):
        try:
            func = self.observables[name]
        except KeyError:
            raise ValueError(f"Unknown observable: {name}")
        return func(lattice, workingLattice)
