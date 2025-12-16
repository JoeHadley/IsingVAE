import numpy as np
import os
import matplotlib.pyplot as plt
from Simulation import Simulation
from Lattice import Lattice, SquareND
from UpdateProposer import *
from Observer import *
from Action import *


latdims = [8,8]


myAction = Action()
myLattice = SquareND(latdims)
#myUpdateProposer = VAEProposer(input_dim=4, latent_dim=2, double_input=False, learning=True, device='cpu', MCbeta=1.0, VAEbeta=1.0)

distribution = "uniform"  # Options: "uniform", "gaussian"

myUpdateProposer = MetropolisProposer(distribution=distribution)
myObserver = Observer("phiBar")
mySimulation = Simulation( myLattice, myAction, myUpdateProposer, myObserver)

mySimulation.updateCycles(1000)

data = mySimulation.acceptanceRateHistory[:mySimulation.acceptanceRateHistoryCount]

#Histogram of acceptance rates
print(mySimulation.acceptanceRateHistoryCount)
plt.hist(data, bins=100)
plt.title("Histogram of Acceptance Rates\n" + "8x8, " + str.capitalize(distribution) + " Distribution")
#plt.plot(data)


plt.show()