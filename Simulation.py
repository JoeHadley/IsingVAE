
import numpy as np
from Lattice import Lattice, Square2D
from UpdateProposer import UpdateProposer
from Action import Action

class Simulation:
    def __init__(self, Lattice,Action,UpdateProposer):
        self.Lattice = Lattice
        self.Action = Action
        self.UpdateProposer = UpdateProposer

        # Initialize lattice properties
        self.latdims = self.Lattice.latdims
        self.Ntot = np.prod(self.latdims)
        self.dim = len(self.latdims)
        self.workingLattice = Lattice.lat

        self.addressList = np.arange(self.Ntot)



    
    def updateCycles(self,cycles):
        for c in range(cycles):
            self.updateCycle()
    
    def updateCycle(self):
        self.UpdateProposer.updateCycle(self,Lattice,Action)

    def initialize(self, initConfig=None):
        self.lat = self.Lattice.initializeLattice(initConfig)


Square2D = Square2D(latdims=(10, 10), shuffle=True)

simulation = Simulation(Lattice, Action, UpdateProposer)

#def recordObservable(self,value = None):
#if value == None:
#    value = self.observableFunc()

#if self.historyCount < self.historyLimit:
#    self.history[self.historyCount] = value
#    self.historyCount += 1
#else:
#    print("Observable History Limit Reached")
#    self.historyLimitReached = True # Just warn once

#def returnHistory(self):
#    return self.history[0:self.historyCount]

#self.historyLimit = historyLimit
#self.history = np.zeros(historyLimit)
#self.historyCount = 0
#self.historyLimitReached = False



#self.recording = observableFuncName is not None
#self.observableFuncName = observableFuncName
#self.observableFunc = self.observables.get(self.observableFuncName, self.observables["empty"])
#self.recordWhileWarming = recordWhileWarming if self.recording else False



#self.observables = {
#    "phi4": lambda: self.expectation(func=lambda x: x**4),
#    "phi2": lambda: self.expectation(func=lambda x: x**2),
#    "action": self.findAction,
#    "phiBar": self.expectation,  # Default expectation#

#    "empty": lambda: 0,
#}
