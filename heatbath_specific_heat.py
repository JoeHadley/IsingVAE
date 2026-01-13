import numpy as np
from Lattice import Lattice, SquareND
from UpdateProposer import *
import windowing
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
sideLength = 4
latdims = np.array([sideLength] * dim)
lat1 = SquareND(latdims, shuffle=True)
lat2 = SquareND(latdims, shuffle=True)
act1 = Action(m=1.0)
act2 = Action(m=1.0)
upd1 = MetropolisProposer()
upd2 = VAEProposer( lattice_dim=dim,
                              window_side_length=2,
                              latent_dim=1,
                              double_input=True,
                              learning = False,
                              batch_size=1,
                              device='cpu',
                              VAEbeta=1.0,
                              debug=False)
obs1 = Observer("action")
obs2 = Observer("action")
sim1 = Simulation(
    lattice=lat1,
    action=act1,
    updateProposer=upd1,
    observer=obs1,
    warmCycles=0
    )

sim2 = Simulation(
    lattice=lat2,
    action=act2,
    updateProposer=upd2,
    observer=obs2,
    warmCycles=0
    )

sim1.workingLattice = np.random.uniform(-1, 1, size=lat1.Ntot)
sim2.workingLattice = np.random.uniform(-1, 1, size=lat2.Ntot)

total_cycles = 100
check_each = 10

for i in range(total_cycles):
    if i % check_each == 0:
        print(f"Metropolis Cycle {i}/{total_cycles}")
    sim2.updateCycle()

sim1.updateCycles(1000)
#for i in range(100):
#    print(f"VAE Cycle {i*100}/1000")
#    sim2.updateCycles(100)




specific_heat1, heat_error1 = calculate_specific_heat(sim1)
print("Metropolis Specific Heat per site:", specific_heat1, " ± ",heat_error1)

specific_heat2, heat_error2 = calculate_specific_heat(sim2)
print("VAE Specific Heat per site:", specific_heat2, " ± ",heat_error2)

accepted = np.sum(sim2.acceptanceHistory[:sim2.acceptanceRateHistoryCount])
total = sim2.acceptanceRateHistoryCount
acceptance_rate = accepted / total if total > 0 else 0

print(f"Acceptance Rate: {acceptance_rate:.4f}")
# print average acceptance probability from history

print(sim2.acceptanceHistory[:sim2.acceptanceRateHistoryCount].mean())

#plot acceptance probability history
import matplotlib.pyplot as plt
plt.plot(sim2.acceptanceHistory[:sim2.acceptanceRateHistoryCount])
plt.xlabel("Proposal Number")
plt.ylabel("Acceptance (1) or Rejection (0)")
plt.title("VAE Proposal Acceptance History")
plt.savefig("vae_acceptance_history.png")


