import numpy as np
from Lattice import Lattice, SquareND
from UpdateProposer import *

from Action import Action
from Observer import Observer
from VAEDefinition import VAE
from ReaderWriter import ReaderWriter
from statFunctions import jackknife_bins, integrated_autocorr_time
from Simulation import Simulation



dim = 1
sideLength = 3
latdims = np.array([sideLength] * dim)
myLattice = SquareND(latdims, shuffle=True)
myAction = Action(m=1.0)
myUpdateProposer=VAEProposer( lattice_dim=dim, window_size=3, latent_dim=1, double_input=True, batch_size=10, device='cpu', beta=1.0)

my_simulation = Simulation(
    beta=1.0,
    MyLattice=myLattice,
    MyAction=myAction,
    MyUpdateProposer=myUpdateProposer,
    MyObserver=Observer("phiBar"),
    warmCycles=0
    )

print("Lattice double_input:", myUpdateProposer.double_input)
print("Simulation double_input:", my_simulation.updateProposer.double_input)


my_simulation.showLattice()
# Optional argument set to False to indicate we do not want to use learning


# Learning, Double Input
my_simulation.updateCycles(5,optional_arg=True)
my_simulation.showLattice()
# Not Learning, Double Input
my_simulation.updateCycles(5,optional_arg=False)
my_simulation.showLattice()


myUpdateProposer.double_input = False
# Learning, Single Input
my_simulation.updateCycles(5,optional_arg=True)
my_simulation.showLattice()
# Not Learning, Single Input
my_simulation.updateCycles(5,optional_arg=True)
my_simulation.showLattice()

my_simulation.showLattice()

