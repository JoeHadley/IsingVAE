import numpy as np
from Lattice import Lattice, SquareND
from UpdateProposer import MetropolisProposer

from Action import Action
from Observer import Observer
#from VAEDefinition import VAE
from ReaderWriter import ReaderWriter
from statFunctions import jackknife_bins, integrated_autocorr_time
from Simulation import Simulation



dim = 2
sideLength = 8
latdims = np.array([sideLength] * dim)
myLattice = SquareND(latdims, shuffle=True)
myAction = Action(m=1.0)

myUpdateProposer=MetropolisProposer()

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
my_simulation.updateCycles(1000)
b = my_simulation.workingLattice.copy()
