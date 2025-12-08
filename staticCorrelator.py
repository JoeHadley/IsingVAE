import numpy as np
import os
import matplotlib.pyplot as plt
from Simulation import Simulation
from Lattice import Lattice, SquareND
from UpdateProposer import *
from Observer import *
from Action import *


latdims = [10]


myAction = Action()
myLattice = SquareND(latdims)
myUpdateProposer = MetropolisProposer()
myObserver = Observer("phiBar")
mySimulation = Simulation(1.0, myLattice, myAction, myUpdateProposer, myObserver,1000)

mySimulation.updateCycles(50)

plt.plot(mySimulation.workingLattice)
plt.show()