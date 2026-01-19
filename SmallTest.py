import numpy as np
from Lattice import Lattice, SquareND
from UpdateProposer import *
import windowing
from Action import Action
from Observer import Observer
from ReaderWriter import ReaderWriter
from statFunctions import jackknife_bins, integrated_autocorr_time
from Simulation import Simulation


dim = 2
sideLength = 4
latdims = np.array([sideLength] * dim)


window_side_length = 1

lat = SquareND(latdims, shuffle=True)
act = Action(m=1.0)
upd = VAEProposer( lattice_dim=dim,
                              window_side_length=window_side_length,
                              latent_dim=1,
                              double_input=True,
                              learning = False,
                              batch_size=1,
                              device='cpu',
                              VAEbeta=1.0,
                              debug=True)
obs = Observer("action")

sim = Simulation(
    lattice=lat,
    action=act,
    updateProposer=upd,
    observer=obs,
    warmCycles=0
    )

sim.workingLattice = np.random.uniform(-1, 1, size=lat.Ntot)
site = 0
window, latdims = lat.createWindow(old_lattice=sim.workingLattice,site=site, window_size=window_side_length)
print("Site",site,", Created window:", window.reshape(latdims))