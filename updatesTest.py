import numpy as np
import random as r
import matplotlib.pyplot as plt
from analyticSolution import *
from Lattice import Lattice, SquareND
import time

from Lattice import Lattice, SquareND
from Action import Action
from Simulation import Simulation
from UpdateProposer import MetropolisProposer, HeatbathProposer, DummyProposer
from ReaderWriter import ReaderWriter
import os
from Observer import Observer


latdims = (2,2)

Lattice = SquareND(latdims)
Action = Action()
UpdateProposer = HeatbathProposer( )

sim = Simulation(1.0,Lattice,Action,UpdateProposer,None)

#sim.workingLattice = np.array([0,0,0,0,0,0,0,0,0])

sim.showLattice()

sim.updateCycles(10)

sim.showLattice()