import numpy as np
from Lattice import Lattice, SquareND
from UpdateProposer import *

from Action import Action
from Observer import Observer
from VAEDefinition import VAE
from ReaderWriter import ReaderWriter
from statFunctions import jackknife_bins, integrated_autocorr_time
from Simulation import Simulation



dim = 2
sideLength = 3
latdims = np.array([sideLength] * dim)
myLattice = SquareND(latdims, shuffle=True)
myAction = Action(m=1.0)
myUpdateProposer=VAEProposer( lattice_dim=dim, 
                              window_side_length=2, 
                              latent_dim=1, 
                              double_input=False, 
                              learning = False, 
                              batch_size=1, 
                              device='cpu', 
                              VAEbeta=1.0)

my_simulation = Simulation(
    lattice=myLattice,
    action=myAction,
    updateProposer=myUpdateProposer,
    observer=Observer("phiBar"),
    warmCycles=0
    )

my_simulation.workingLattice = np.random.uniform(-1, 1, size=myLattice.Ntot)

a = my_simulation.workingLattice.copy()

my_simulation.showLattice()

# Learning, Double Input
my_simulation.updateCycles(10)
b = my_simulation.workingLattice.copy()

my_simulation.showLattice()

print(np.reshape(a-b, (sideLength, sideLength)))