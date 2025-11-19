import numpy as np
import random as r
import math
import matplotlib.pyplot as plt
from UpdateProposer import ToyMVAEProposer
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

        # Using shift2 for verification
        shifted_site2 = lat.shift2(site, direction, jump)%lat.Ntot

        

        if shifted_site != shifted_site2:
            print(f"Site {site}, direction {direction}, jump {jump}: {shifted_site} (shift) vs {shifted_site2} (shift2). MISMATCH!")
        else:
            print(f"Site {site}, direction {direction}, jump {jump}: {shifted_site} (shift) vs {shifted_site2} (shift2). MATCH!")