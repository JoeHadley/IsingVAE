import numpy as np
from Lattice import Lattice, SquareND
from UpdateProposer import *

from Action import Action
from Observer import Observer
from VAEDefinition import VAE
from ReaderWriter import ReaderWriter
from statFunctions import jackknife_bins, integrated_autocorr_time
from Simulation import Simulation



dim = 3
sideLength = 3
latdims = np.array([sideLength] * dim)
my_lattice = SquareND(latdims, shuffle=True)

my_simulation = Simulation(
    beta=1.0,
    MyLattice=my_lattice,
    MyAction=Action(m=1.0),
    MyUpdateProposer=HeatbathProposer(beta=1.0, shuffle=True),
    warmCycles=0
)


init_config = np.arange(my_lattice.Ntot)
print(init_config.reshape(latdims))


def createWindow(simulation, site, window_size):
        # Same dimensions as wider lattice
    latdims = simulation.lattice.latdims
    dim = len(latdims)

    Ntot = simulation.lattice.Ntot
    ntot = window_size**dim
    new_array = np.zeros(ntot)

    for i in range(ntot):
        number = np.base_repr(i, base=window_size).zfill(dim)
        moving_index = site
        for d in range(dim):
            
            digit = int(number[d])
            moving_index = simulation.lattice.shift(moving_index,d , digit)

        new_array[i] = init_config[moving_index]
        
    
    return new_array

workingDimension = [3,2,1,0]

#    for k in range(window_size):
#        kWorkingDimension = 3
#        k_site = simulation.lattice.shift(site, dim-kWorkingDimension,k)
#        k_digit = k*window_size**(kWorkingDimension-1)


#        for j in range(window_size):
#            jWorkingDimension = kWorkingDimension - 1
#            j_site = simulation.lattice.shift(site, dim-jWorkingDimension,j)
#            j_digit = j*window_size**(jWorkingDimension-1)
#            for i in range(window_size):
#                i_digit = i*window_size**0
#                new_array[k_digit+j_digit+i_digit] = simulation.lattice.shift(j_site, 2,i)
    

window = createWindow( my_simulation, site=5, window_size=2)
print("Created window:")
print(window)