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
my_action = Action(m=1, l=0)
my_action.printParams()
#my_proposer = MetropolisProposer(dMax=1, beta=5)

input_dim = 3  # Example input dimension
hidden_dim = 2  # Example hidden dimension
latent_dim = 1  # Example latent dimension


beta = 5.0
my_proposer = ToyMVAEProposer()

#my_proposer2 = MetropolisProposer(dMax=1, beta=beta, shuffle=True)

my_observer = Observer(observableFuncName= "phiBar", historyLimit=10000)


simulation = Simulation(beta, my_lattice, my_action, my_proposer, my_observer,warmCycles=100)
simulation.updateCycles(cycles=100)
print("Here's the history of the observable:")
print(simulation.observer.returnHistory())




energies = simulation.observer.returnHistory()
Ntot = simulation.Ntot  # or number of lattice sites, depending on your definition


# Histogram of energies
import matplotlib.pyplot as plt
plt.hist(energies, bins=30)
plt.xlabel('Energy')
plt.ylabel('Probability Density')
plt.title('Histogram of Energy Measurements')
plt.show()
