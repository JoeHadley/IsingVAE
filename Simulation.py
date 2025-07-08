
import numpy as np
from Lattice import Lattice, Square2D
from UpdateProposer import UpdateProposer, MetropolisProposer
from Action import Action
from Observer import Observer

class Simulation:
    def __init__(self, MyLattice,MyAction,MyUpdateProposer,MyObserver=None):
        self.lattice = MyLattice
        self.action = MyAction
        self.updateProposer = MyUpdateProposer
        self.observer = MyObserver

        # Take lattice properties from the lattice object
        self.latdims = self.lattice.latdims
        self.Ntot = self.lattice.Ntot
        self.dim = len(self.latdims)
        
        self.workingLattice = self.lattice.lat.copy()  # Working lattice for updates

        self.addressList = np.arange(self.Ntot)


    def showLattice(self):
        showLat = np.reshape(self.workingLattice,self.latdims)
        print(showLat)
    
    def updateCycles(self,cycles):
        for c in range(cycles):
            print(c)
            self.updateCycle()
    
    def updateCycle(self):
        for n in range(self.Ntot):
            self.updateProposer.update(self, site=n)
            if self.observer is not None:
                self.observer.recordObservable(self.action, value=self.workingLattice[n])

    def initialize(self, initConfig=None):
        self.lat = self.lattice.initializeLattice(initConfig)
    
    def saveConfig(self, filename):
        with open(filename, 'w') as f:
            for value in self.workingLattice:
                f.write(f"{value}\n")  



my_lattice = Square2D(latdims=(10, 10), shuffle=True)
my_action = Action(m=1, l=1, dMax0=1)
my_action.printParams()
my_proposer = MetropolisProposer(dMax=1, beta=5)
my_proposer.printParams()
my_observer = Observer(observableFuncName="phiBar", recordWhileWarming=True, historyLimit=10000)
simulation = Simulation(my_lattice, my_action, my_proposer, my_observer)
simulation.initialize(initConfig=None)
simulation.updateCycles(cycles=10)
simulation.showLattice()
simulation.observer.returnHistory()


import matplotlib.pyplot as plt

# Reshape the lattice
showLat = np.reshape(simulation.workingLattice, simulation.latdims)

# Create a figure with 1 row and 2 columns
fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # Increase width for side-by-side

# Plot lattice configuration
im = axs[0].imshow(showLat, cmap='Greys', origin='lower', aspect='equal')
fig.colorbar(im, ax=axs[0], label="Value")
axs[0].set_title("φ⁴ Lattice Configuration")
axs[0].set_xticks([])
axs[0].set_yticks([])

# Plot histogram
axs[1].hist(simulation.workingLattice, bins=30, density=True, alpha=0.7, color='blue')
axs[1].set_title("Histogram of Lattice Values")
axs[1].set_xlabel("Lattice Value")
axs[1].set_ylabel("Density")
axs[1].grid(True)

plt.tight_layout()
plt.show()
