from dataclasses import dataclass 
from typing import Optional
import numpy as np

@dataclass
class foobar:
  a: int
  b: int
  c: int

  def readout(self):
    print(f"{self.a},{self.b},{self.c}")
  def readout2(self):
    print(f"{self.a},{self.b},{self.c}")

f = foobar(1,2,3)
f.readout()
f.readout2()


from Lattice import Lattice, SquareND
from Action import Action
from UpdateProposer import *
from Observer import Observer
from ReaderWriter import *


@dataclass
class Simulation2:
  #def __init__(self, beta, lattice,action,updateProposer,observer=None,warmCycles=0,shuffle_list=True, initConfig = None):
  beta: float
  lattice: Lattice 
  action: Action
  updateProposer: UpdateProposer
  observer: Observer

  warmCycles: int = 0
  shuffle_list: bool = True
  initConfig: Optional[np.ndarray] = None
  

  def __post_init__(self):
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
    self.shuffle_list = True

    self.workingLattice = self.lattice.initializeLattice(self.initConfig)
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

    print(f"Site: {site}, old value: {self.workingLattice[site]}, new value: {new_lattice[site]}")

    roll = np.random.uniform(0,1)

    if roll <= acceptance_probability:
      self.workingLattice = new_lattice




  
  def saveConfig(self, filename):
    self.ReaderWriter.writeConfig(self, filename)



latdims = (2,2)
lat = SquareND(latdims)
action = Action()
updateProposer = DummyProposer()
observer = Observer("phiBar")
sim = Simulation2(1.0,lat,action,updateProposer,None,0,True)