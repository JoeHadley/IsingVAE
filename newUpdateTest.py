from Simulation import Simulation
from Lattice import Lattice, SquareND
import numpy as np
from UpdateProposer import *
from Action import Action
from Observer import Observer
from ReaderWriter import ReaderWriter
from statFunctions import jackknife_bins, integrated_autocorr_time
from VAEDefinition import VAE




Lattice = SquareND(np.array([4,4]), shuffle=True)
Action = Action(m=1.0)
Updater = DummyProposer()
Observer = Observer("empty")
Sim = Simulation(
    lattice=Lattice,
    action=Action,
    updateProposer=Updater,
    warmCycles=0
    )

Sim.workingLattice = np.random.uniform(-1, 1, size=Lattice.Ntot)
Sim.showLattice()
Sim.sim_update()
Sim.showLattice()