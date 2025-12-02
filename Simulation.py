
import numpy as np
from Lattice import Lattice, SquareND
from UpdateProposer import *

from Action import Action
from Observer import Observer
from VAEDefinition import VAE
from ReaderWriter import ReaderWriter
from statFunctions import jackknife_bins, integrated_autocorr_time
class Simulation:
    def __init__(self, beta, lattice,action,updateProposer,observer=None,warmCycles=0):
        self.beta = beta
        self.lattice = lattice
        self.action = action
        self.updateProposer = updateProposer
        self.observer = observer

        self.warmCycles = warmCycles

        self.ReaderWriter = ReaderWriter()

        # Prepare array for acceptance rates
        self.historyLimit = 10000
        self.history = np.zeros(self.historyLimit)
        self.historyCount = 0
        self.historyLimitReached = False


        # Take lattice properties from the lattice object
        self.latdims = self.lattice.latdims
        self.Ntot = self.lattice.Ntot
        self.dim = len(self.latdims)
        self.initialize()



    def showLattice(self):
        showLat = np.reshape(self.workingLattice,self.latdims)
        print(showLat)
    
    def updateCycles(self,cycles,warmingFlag=False):
      for c in range(cycles):
        self.updateCycle(warmingFlag)
           
    def updateCycle(self, warmingFlag=False):
        self.updateProposer.updateCycle(self)
        if not warmingFlag and self.observer is not None:
            self.observer.recordObservable(self)

    def sim_update(self):
        acceptance_probability = self.updateProposer
        newLattice, acceptance_probability = self.updateProposer.update(self.workingLattice)

        roll = np.random.uniform(0,1)

        if roll <= acceptance_probability:
          self.workingLattice = newLattice




    def initialize(self, initConfig=None):
        self.lat = self.lattice.initializeLattice(initConfig)
        self.workingLattice = self.lat.copy()
        self.updateCycles(self.warmCycles, warmingFlag=True)

    
    def saveConfig(self, filename):
        self.ReaderWriter.writeConfig(self, filename)










