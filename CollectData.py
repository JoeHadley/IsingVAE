import numpy as np
from Lattice import Lattice, SquareND
from UpdateProposer import MetropolisProposer

from Action import Action
from Observer import Observer
#from VAEDefinition import VAE
from ReaderWriter import ReaderWriter
from statFunctions import jackknife_bins, integrated_autocorr_time
from Simulation import Simulation



dim = 2
sideLength = 8
latdims = np.array([sideLength] * dim)
myLattice = SquareND(latdims, shuffle=True)
myAction = Action(m=1.0)
myReaderWriter = ReaderWriter()

myUpdateProposer=MetropolisProposer()

my_simulation = Simulation(
    lattice=myLattice,
    action=myAction,
    updateProposer=myUpdateProposer,
    observer=Observer("phiBar"),
    readerWriter=myReaderWriter,
    warmCycles=0
    )

config_directory = "Configs/"
filename = "8x8_configs.bin"
filestring = config_directory + filename

my_simulation.workingLattice = np.random.uniform(-1, 1, size=myLattice.Ntot)
my_simulation.updateCycles(1000)
for i in range(1000):
  my_simulation.updateCycles(1000)
  my_simulation.saveConfig(filestring)
