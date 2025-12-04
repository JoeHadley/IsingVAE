
import numpy as np
from Lattice import Lattice, SquareND
from UpdateProposer import *

from Action import Action
from Observer import Observer
from VAEDefinition import VAE
from ReaderWriter import ReaderWriter
from statFunctions import jackknife_bins, integrated_autocorr_time
class Simulation:
  def __init__(self, beta, lattice,action,updateProposer,observer=None,warmCycles=0,shuffle_address_list=True, initConfig = None):
    self.beta = beta
    self.lattice = lattice
    self.action = action
    self.updateProposer = updateProposer
    self.observer = observer

    self.warmCycles = warmCycles

    self.ReaderWriter = ReaderWriter()

    # Prepare array for acceptance rates
    self.acceptanceRateHistoryLimit = 10000
    self.acceptanceRateHistory = np.zeros(self.acceptanceRateHistoryLimit)
    self.acceptanceRateHistoryCount = 0
    self.acceptanceRateHistoryLimitReached = False


    # Take lattice properties from the lattice object
    self.latdims = self.lattice.latdims
    self.Ntot = self.lattice.Ntot
    self.dim = len(self.latdims)


    self.address_list = np.arange(self.Ntot)
    self.cycleSize = self.Ntot # Default cyclesize
    self.shuffle_address_list = shuffle_address_list

    self.workingLattice = self.lattice.initializeLattice(initConfig)
    self.updateCycles(self.warmCycles, warmingFlag=True)



  def shuffle_address_list(self):
    self.address_list = np.random.permutation(self.address_list)


  def showLattice(self):
      showLat = np.reshape(self.workingLattice,self.latdims)
      print(showLat)
  
  def updateCycles(self,cycles,warmingFlag=False):
    for c in range(cycles):
      self.updateCycle(warmingFlag)
          
  def updateCycle(self, warmingFlag=False):
    if self.shuffle_list:
      self.shuffle_address_list()

    for i in range(self.cycleSize):
      site = self.address_list[i]
  

      self.update(site)

    if not warmingFlag and self.observer is not None:
      self.observer.recordObservable(self)

  def update(self,site):
    

    new_lattice,acceptance_probability = self.updateProposer.propose(self,site)

    roll = np.random.uniform(0,1)

    if roll <= acceptance_probability:
      self.workingLattice = new_lattice
    


    if not self.acceptanceRateHistoryLimitReached:
      self.acceptanceRateHistory[self.acceptanceRateHistoryCount] = acceptance_probability
      self.acceptanceRateHistoryCount += 1
      if self.acceptanceRateHistoryCount == self.acceptanceRateHistoryLimit:
        self.acceptanceRateHistoryLimitReached = True
    





  
  def saveConfig(self, filename):
    self.ReaderWriter.writeConfig(self, filename)










