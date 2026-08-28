import numpy as np
import random as r
import math

import matplotlib.pyplot as plt
from UpdateProposer import ToyMVAEProposer
from Lattice import SquareND
from Simulation import Simulation
from Action import Action
from Observer import Observer


class Lattice2:


def shift2(self, site, dim, jump):
    stride = int(self.vec[dim+1])    # size of next dimension in flat indexing
    size   = int(self.latdims[dim])  # number of sites in this dimension

    # extract coordinate in this dimension
    coord = (site // stride) % size

    # shift with periodic boundaries
    coord = (coord + jump) % size

    # put back into flat index
    new_site = site - (site // stride % size) * stride + coord * stride
    return new_site


vec = np.ones(dim + 1)*Ntot

for v in range(dim):
    vec[v+1] = vec[v]/latdims[v]


def shift(self,site,dim,jump):



    shiftedSite = int(self.vec[dim]* (site //self.vec[dim]) + (site + jump * self.vec[dim+1] + self.Ntot) % (self.vec[dim]))
    return shiftedSite

