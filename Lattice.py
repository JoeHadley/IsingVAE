import numpy as np
import random as r
import matplotlib.pyplot as plt
import time
import struct
import base64
import os
from abc import ABC, abstractmethod

class Lattice(ABC):
    def __init__(self, latdims, Ntot,lat):
        self.latdims = latdims  # all lattices will have this
        self.Ntot = Ntot
        self.addressList = np.arange(self.Ntot)

    @abstractmethod
    def shift(self, site, dim, jump):
        "Shift a site in the lattice by a given jump in a specified dimension."
        pass



class SquareND(Lattice):
    def __init__(self, latdims, shuffle = False, proposer = None):
        self.latdims = latdims
        self.Ntot = np.prod(self.latdims)
        self.dim = len(self.latdims)
        self.lat = np.zeros(self.Ntot)  # Initialize the lattice with zeros
        super().__init__(latdims, np.prod(latdims), self.lat)
        
        
        self.addressList = np.arange(self.Ntot)
        self.shuffle = shuffle

        self.proposer = proposer

        #self.previousLat = self.lat.copy()
        #self.previousMove = np.zeros(2)

        self.jumps = np.zeros(2,dtype=int)
        self.jumps[0] = -1
        self.jumps[1] = 1
        








        self.vec = np.ones(self.dim + 1)*self.Ntot

        for v in range(self.dim):
            self.vec[v+1] = self.vec[v]/latdims[v]






    def initializeLattice(self, initConfig=None):
        return np.random.normal(0, 1, self.Ntot) if initConfig is None else np.array(initConfig)
    

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
        neighbs = np.array([self.shift(site, dim, jump) for dim in range(self.dim) for jump in self.jumps], dtype=int)
        return neighbs

    




    
    
my_lattice = SquareND(latdims=np.array([3]), shuffle=True)

for i in range(my_lattice.Ntot):
    my_lattice.lat[i] = i

my_lattice.show()

print(my_lattice.shift(0,0,-1))



