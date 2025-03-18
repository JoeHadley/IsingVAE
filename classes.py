import numpy as np
import random as r
import matplotlib.pyplot as plt
import time
import struct
import base64
import os

observables = {
            "phi4": lambda: self.expectation(func=lambda x: x**4),
            "phi2": lambda: self.expectation(func=lambda x: x**2),
            "action": self.findAction,
            "phiBar": self.expectation,  # Default expectation

            "empty": lambda: 0,
        }




# The purpose of this class is to handle the simulation without worrying what the update rule, lattice, or action is. It takes a proposed update, polls the probability based on the action, and then accepts or rejects it
class MonteCarlo:
    def __init__(self,latticeGeometry, latDims, action, updateRule,warmCycles = None, initConfig = None):
        self.lat = latticeGeometry.latticeSetup(latDims,initConfig)



        self.action = action
        self.updateRule = updateRule



        self.previousLat = self.lattice.lat.copy()
        self.previousMove = self.previousMove = np.zeros(2)


        self.warming = True
        if warmCycles is not None:
            self.metroCycles(warmCycles)
        self.warming = False


# The action has the role of calculating the action of a given lattice
class action:
    def __init__(self,m=1,l=0,):
        self.m = m
        self.l = l
    
    def findAction(self, lattice, latticeGeometry):
        S = 0
        dS = 0

        for n in range(self.Ntot):
            dS = 0
            neighbTotal = 0
            neighbours = latticeGeometry.getNeighbours(n)
            for neighbour in neighbours:
                neighbTotal += lattice[neighbour]
            
            dS += (latticeGeometry.dim + (self.m**2)/2)*lattice[n]**2 - lattice[n]*neighbTotal + (self.l / 24) * lattice[n]**4 
            
            S += dS
        return S

    def actionChange(self, lattice, latticeGeometry, address,d):


        neighbours = latticeGeometry.getNeighbours(n)

        

        neighbSum = 0
        for neighbour in neighbours:
            neighbSum += lattice[neighbour]

        dS = d*( 2*lattice[address]*(2+ self.m*self.m/2) - neighbSum  ) \
        + d*d*(2 + self.m*self.m/2 )


        dSl = self.l*( \
        + d*np.power(lattice[address],3)/6  \
        + d*d*np.power(lattice[address],2)/4 \
        + d*d*d*lattice[address]/6 \
        + d*d*d*d/24 )

        return dS +dSl
        

# The observer holds the history
class Observer:
    def __init__(self,historyLimit = int(1e6),observableFuncName = None, recordWhileWarming = False):
        self.historyLimit = historyLimit
        self.history = np.zeros(historyLimit)
        self.historyCount = 0
        self.historyLimitReached = False



        self.recording = observableFuncName is not None
        self.observableFuncName = observableFuncName
        self.observableFunc = self.observables.get(self.observableFuncName, self.observables["empty"])
        self.recordWhileWarming = recordWhileWarming if self.recording else False


class Writer:
    def __init__(self):
        pass


# The update rule is a class whose role is to know how to update a configuration

class updateRule:
    def __init__(self,dMax0 = 1, shuffle = False):
        self.update_rule = updateRule
        self.shuffle = shuffle
        self.dMax = dMax0


    def metroUpdate(self,action,n):



        d = r.gauss(0,self.dMax)

        

        dS = action.actionChange(n,d)

        boltFactor = np.exp(-dS)

        p = min(1,boltFactor)

        roll = r.uniform(0,1)


        # Apply previous move to previousLat
        self.previousLat[int(self.previousMove[0])] += self.previousMove[1]

        # Apply this move to current lattice

        update = 0
        if roll <= p:
            update = d

        self.lat[n] += update
        self.previousMove[0] = n
        self.previousMove[1] = update # Is sometimes 0  
            #self.previousMove[0] = True


        
        

        if self.recording and (not self.warming or self.recordWhileWarming) and not self.historyLimitReached:

            self.recordObservable()


# The latticeGeometry class holds the geometry of the lattice. It holds the logic for naming neighbours, shifting, setting up, and holds the lattice itself
class latticeGeometry:
    def __init__(self,latdims,shiftFunc):
        self.latdims = latdims
        self.Ntot = np.prod(self.latdims)
        self.dim = len(self.latdims)
        self.addressList = np.arange(self.Ntot)




        self.jumps = np.zeros(2,dtype=int)
        self.jumps[0] = -1
        self.jumps[1] = 1




        self.vec = np.ones(self.dim + 1)*self.Ntot

        for v in range(self.dim):
            self.vec[v+1] = self.vec[v]/latdims[v]

        
        neighbTotal = 0
        for d in range(self.dim):
            neighbTotal += self.lattice[self.shift(n,d,1)]
    



        def latticeSetup(self, initConfig):
            return np.random.normal(0, 1, self.Ntot) if initConfig is None else np.array(initConfig)
        
        def shift(self,site,dim,jump):
            shiftedSite = int(self.vec[dim]* (site //self.vec[dim]) + (site + jump * self.vec[dim+1] + self.Ntot) % (self.vec[dim]))
            return shiftedSite
        
        def getNeighbours(self,site):
            neighbs = np.array([self.shift(site, dim, jump) for dim in range(self.dim) for jump in self.jumps], dtype=int)
            return neighbs


        












        