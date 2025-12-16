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

my_lattice = SquareND(latdims, shuffle=True)
my_action = Action(m=1, l=0)
#my_action.printParams()
#my_proposer = MetropolisProposer(dMax=1, beta=5)

input_dim = 3  # Example input dimension
hidden_dim = 2  # Example hidden dimension
latent_dim = 1  # Example latent dimension


warmCycles=10000

# Define proposers and observers in a list
proposers = [
    ("ToyMVAE", ToyMVAEProposer( shuffle=False)),
    ("Heatbath1", HeatbathProposer(shuffle=False)),
    ("Heatbath2", HeatbathProposer(shuffle=False)),
    ("Heatbath3", HeatbathProposer(shuffle=False)),
]

# Create observers
observers = [Observer(observableFuncName="phiBar", historyLimit=100000) for _ in proposers]

# Create simulations
simulations = [
    Simulation( my_lattice, my_action, proposer, observer, warmCycles)
    for (_, proposer), observer in zip(proposers, observers)
]

# Run all simulations
cycles = 10000
for sim in simulations:
    sim.updateCycles(cycles=cycles)

# Print results
for (name, _), obs in zip(proposers, observers):
    phi_history = obs.returnHistory()
    print(f"{name} Results:")
    print(f"  Average phiBar: {np.mean(phi_history)}")
    print(f"  Variance of phiBar: {np.var(phi_history)}\n")






# Histogram of energies
#import matplotlib.pyplot as plt
#plt.hist(energies, bins=30)
#plt.xlabel('Energy')
#plt.ylabel('Probability Density')
#plt.title('Histogram of Energy Measurements')
#plt.show()
