import numpy as np
from Lattice import Lattice, SquareND
from UpdateProposer import *

from Action import Action
from Observer import Observer
from ReaderWriter import ReaderWriter
from statFunctions import jackknife_bins, integrated_autocorr_time
from Simulation import Simulation

def calculate_specific_heat(simulation):
    actions = simulation.observer.returnHistory()
    this_lattice = simulation.lattice
    actions_sq = [a**2 for a in actions]

    mean_action_sq = (np.mean(actions))**2
    mean_sq_action = np.mean(actions_sq)

    specific_heat = (mean_sq_action - mean_action_sq)/(this_lattice.Ntot)
    heat_error = np.std(actions_sq - mean_action_sq)/(this_lattice.Ntot * np.sqrt(len(actions)))

    return specific_heat, heat_error

dim = 2
sideLength = 5
latdims = np.array([sideLength] * dim)
lat1 = SquareND(latdims, shuffle=True)
lat2 = SquareND(latdims, shuffle=True)
act1 = Action(m=1.0)
act2 = Action(m=1.0)
upd1 =HeatbathProposer()
upd2 = VAEProposer( lattice_dim=dim, window_size=2, latent_dim=1, double_input=False, batch_size=1, device='cpu', beta=1.0)
obs1 = Observer("action")
obs2 = Observer("action")
sim1 = Simulation(
    beta=1.0,
    lattice=lat1,
    action=act1,
    updateProposer=upd1,
    observer=obs1,
    warmCycles=0
    )

sim2 = Simulation(
    beta=1.0,
    lattice=lat2,
    action=act2,
    updateProposer=upd2,
    observer=obs2,
    warmCycles=0
    )

sim1.workingLattice = np.random.uniform(-1, 1, size=lat1.Ntot)
sim2.workingLattice = np.random.uniform(-1, 1, size=lat2.Ntot)
sim1.updateCycles(10000)
for i in range(10):
    print(f"VAE Cycle {i*1000}/100000")
    sim2.updateCycles(1000,optional_arg=False)



specific_heat1, heat_error1 = calculate_specific_heat(sim1)
print("Sim 1 Specific Heat per site:", specific_heat1, " ± ",heat_error1)

specific_heat2, heat_error2 = calculate_specific_heat(sim2)
print("Sim 2 Specific Heat per site:", specific_heat2, " ± ",heat_error2)