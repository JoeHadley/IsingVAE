import numpy as np
import random as r
from dataclasses import dataclass
import math
from abc import ABC, abstractmethod
from Simulation import Simulation
from Action import Action
from UpdateProposer import *
from Lattice import *
from Observer import *


myUpdateProposer = HeatbathProposer()
myLattice = SquareND([4,4])
myAction = Action(m=1.0,l=0.0)
mySimulation = Simulation(lattice=myLattice, action=myAction, updateProposer=myUpdateProposer, warmCycles=0)
mySimulation.updateCycles(100)

print(mySimulation.acceptanceRateHistory[:mySimulation.acceptanceRateHistoryCount])

# plot acceptance rate history
import matplotlib.pyplot as plt
plt.plot(mySimulation.acceptanceRateHistory[:mySimulation.acceptanceRateHistoryCount])
plt.xlabel('Update')
plt.ylabel('Acceptance Rate')
plt.title('Acceptance Rate History')
plt.show()