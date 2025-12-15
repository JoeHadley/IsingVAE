import numpy as np
import random as r
import math
import matplotlib.pyplot as plt


from UpdateProposer import *
from Lattice import SquareND
from Simulation import Simulation
from Action import Action
from Observer import Observer


latdims = np.array([4,3,5])

#for dim in range(1,5):

dim = 3

lat = SquareND(np.array(latdims))



for site in range(lat.Ntot):
    lat.lat[site] = site

lat.show()


for jump in range(1,4):

    for direction in range(dim):

        latdims = np.array([4]*dim)
        my_lattice = SquareND(latdims)

        site = 3
        shifted_site = lat.shift(site, direction, jump)%lat.Ntot

     
        

        print(f"Site {site}, direction {direction}, jump {jump}: {shifted_site} (shift)")
