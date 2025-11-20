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
my_lattice = SquareND(latdims, shuffle=True)

my_simulation = Simulation(
    beta=1.0,
    MyLattice=my_lattice,
    MyAction=Action(m=1.0),
    MyUpdateProposer=VAEProposer( lattice_dim=dim, window_size=3, latent_dim=1, double_input=True, batch_size=10, device='cpu', beta=1.0),
    MyObserver=Observer("phiBar"),
    warmCycles=0
    )

my_simulation.showLattice()
my_simulation.updateCycles(5,learning=False)


