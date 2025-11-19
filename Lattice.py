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
        



        self.strides = np.ones(self.dim + 1)*self.Ntot


        def compute_stride(latdims):
            dim = len(latdims)
            stride = [1]*dim
            for d in range(dim-2, -1, -1):
                stride[d] = stride[d+1] * latdims[d+1]
            return np.array(stride, dtype=int)
        
        self.stride = compute_stride(latdims)


        self.vec = np.ones(self.dim + 1)*self.Ntot

        for v in range(self.dim):
            self.vec[v+1] = self.vec[v]/latdims[v]






    def initializeLattice(self, initConfig=None):
        return np.random.normal(0, 1, self.Ntot) if initConfig is None else np.array(initConfig)
    
    def testMode(self):
        self.lat = np.arange(self.Ntot)

    def scrambleLattice(self):
        for n in range(self.Ntot):
            self.lat[n] = r.gauss(0,1)


    def show(self):
        showLat = np.reshape(self.lat,self.latdims)
        print(showLat)
    
    def shuffleList(self):
        
        indices = np.random.permutation(self.Ntot)
        self.addressList = self.addressList[indices]
    


    def shiftOld(self,site,dim,jump):
        shiftedSite = int(self.vec[dim]* (site //self.vec[dim]) + (site + jump * self.vec[dim+1] + self.Ntot) % (self.vec[dim]))
        return shiftedSite
    

    def shift(self, site, dim, jump):
        stride = int(self.stride[dim])    # size of next dimension in flat indexing
        size   = int(self.latdims[dim])  # number of sites in this dimension

        # extract coordinate in this dimension
        coord = (site // stride) % size

        # shift with periodic boundaries
        coord = (coord + jump) % size

        # put back into flat index
        new_site = site - (site // stride % size) * stride + coord * stride
        return new_site
    





    def getNeighbours(self,site):
        neighbs = np.array([self.shift(site, dim, jump) for dim in range(self.dim) for jump in self.jumps], dtype=int)
        return neighbs
    
    

    def createWindow(self, site, window_size):
        
        # Same dimensions as wider lattice
        dim = len(self.latdims)
        window_dims = np.array([window_size]*dim)
        ntot = window_size**dim
        
        # Get coordinates of the site
        coords = np.zeros(dim, dtype=int)
        s = site
        for d in range(dim):
            coords[d] = (s // self.stride[d]) % self.latdims[d]
            s = site%self.stride[d]

        # Set up output
        out = np.zeros(ntot)


        
        for idx in range(ntot): 

            offset = idx
            new_coords = coords.copy()
            for d in reversed(range(dim)):
                step = offset % window_size
                
                offset //= window_size
                new_coords[d] = (coords[d] + step) % self.latdims[d]
            
            flat = 0
            for d in range(dim):
                flat += new_coords[d] * self.stride[d]
            out[idx] = self.lat[flat]
        
        return out, window_dims


#        new_array = np.zeros(ntot)
#        for i in range(ntot):
#            number = np.base_repr(i, base=window_size).zfill(dim)
#            moving_index = site
#            for d in range(dim):
#                digit = int(number[d])
#                moving_index = self.shift(moving_index,d , digit)
#            new_array[i] = self.lat[moving_index]       
#        return new_array, window_dims


latdims = np.array([3,3,3])

#for dim in range(1,5):

dim = 3

lat = SquareND(np.array(latdims))
lat.testMode()
lat.show()

print(lat.stride)
print(lat.vec)
window , window_dims = lat.createWindow(site=7, window_size=2)

show_window = np.reshape(window,window_dims)
print(show_window)
