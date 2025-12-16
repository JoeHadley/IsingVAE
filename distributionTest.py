import numpy as np
from Lattice import Lattice, SquareND
from UpdateProposer import *

from Action import Action
from Observer import Observer
from ReaderWriter import ReaderWriter
from statFunctions import jackknife_bins, integrated_autocorr_time
from Simulation import Simulation

latdims = (8,8)
lat = SquareND(latdims)
action = Action(m=1.0)

updateProposer = MetropolisProposer(dMax=1.0)

sim = Simulation(
    lattice=lat,
    action=action,
    updateProposer=updateProposer,
    observer=Observer("action"),
    warmCycles=1000
    )

sim.updateCycles(1000)
 
 
