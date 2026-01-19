import numpy as np

class Observer:
  def __init__(self, observableFuncName=None, historyLimit=10000,latdims=None):
    self.observableFuncName = observableFuncName
    self.recording = observableFuncName is not None



    self.observables = {
      "phi4":           (1,       lambda simulation: self.expectation(simulation, func=lambda x: x**4)),
      "phi2":           (1,       lambda simulation: self.expectation(simulation, func=lambda x: x**2)),
      "action":         (1,       lambda simulation: simulation.action.findAction(simulation)),
      "phiBar":         (1,       lambda simulation: self.expectation(simulation, None)),  # Default expectation
      "specificHeat":   (1,       lambda simulation: 0),  # Placeholder
      "empty":          (1,       lambda: 0),
      "Correlator":     (1,       lambda simulation: self.correlator(simulation)),
      "StructFactor":   (latdims, lambda simulation: self.structureFactor(simulation)),
      "sqrtJTJ":            ( 1, lambda simulation: self.computeJTJ(simulation) )
      }


    shape = self.observables[observableFuncName][0]

    self.historyLimit = historyLimit
    if shape == 1 or shape == (1,):
      # scalar history
      self.history = np.zeros(self.historyLimit)
    else:
      # array history
      self.history = np.zeros((self.historyLimit, *shape))

    self.historyCount = 0
    self.historyLimitReached = False




  def expectation(self, simulation, func=None):
    # If func is None, use identity
    lattice = simulation.lattice
    workingLattice = simulation.workingLattice
    Ntot = simulation.Ntot

    if func is None:
      func = lambda x: x

    M = 0
    for n in range(Ntot):
      M += func(workingLattice[n])

    return M / simulation.Ntot
  
  def correlator(self, simulation):
    """
    Compute the nearest-neighbor correlator in an N-dimensional lattice.
    Uses the lattice's `getNeighbours(site)` method.
    """
    lattice = simulation.lattice  # instance of SquareND
    workingLattice = simulation.workingLattice
    Ntot = simulation.Ntot

    total = 0
    z = 0

    for n in range(Ntot):
      neighbors = lattice.getNeighbours(n)  # array of neighbor indices
      z = len(neighbors)
      for m in neighbors:
        total += workingLattice[n] * workingLattice[m]

    # Normalize by total number of interactions
    return total / (Ntot * z)

  def computeObservable(self, simulation, name):
    lattice = simulation.lattice
    workingLattice = simulation.workingLattice

    try:
      func = self.observables[name]
    except KeyError:
      raise ValueError(f"Unknown observable: {name}")
    return func(simulation)


  def structureFactor(self, simulation):
    FTLattice = simulation.returnFT()/np.sqrt(simulation.Ntot)
    S = np.abs(FTLattice)**2
    return S

  def computeJTJ(self, simulation):
    vae = simulation.updateProposer.VAE
    trace_JTJ = vae.lastJacobianTermF.item()
    return trace_JTJ 

  def recordObservable(self, simulation, value=None):

    func = self.observables[self.observableFuncName][1]


    if value is None:
      if self.observableFuncName is not None:
        value = func(simulation)

    
    if self.historyCount < self.historyLimit:
      self.history[self.historyCount] = value
      self.historyCount += 1
    else:
      if not self.historyLimitReached:
        print("Observable History Limit Reached")
        self.historyLimitReached = True

  def returnHistory(self):
    if self.historyLimitReached:
      print("Warning: Observable History Limit Reached")
    return self.history[:self.historyCount]
