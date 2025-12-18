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

data = mySimulation.acceptanceRateHistory[:mySimulation.acceptanceRateHistoryCount]
print(data.min(), data.max())

print(mySimulation.acceptedCount/mySimulation.acceptanceRateHistoryCount)


plt.plot(data)
plt.axhline(y=0.0, linestyle=':', linewidth=1)
plt.xlabel('Update')
plt.ylabel('Acceptance Probability')
plt.ylim(-0.20, 1.05)

plt.title('Heatbath Acceptance Rate History')
plt.show()