import numpy as np

class Observer:
    def __init__(self, observableFuncName=None, historyLimit=10000):
        self.observableFuncName = observableFuncName
        self.recording = observableFuncName is not None


        self.historyLimit = historyLimit
        self.history = np.zeros(historyLimit)
        self.historyCount = 0
        self.historyLimitReached = False


        self.observables = {
                "phi4": lambda simulation: self.expectation(simulation, func=lambda x: x**4),
                "phi2": lambda simulation: self.expectation(simulation, func=lambda x: x**2),
                "action": lambda simulation: simulation.action.findAction(simulation),
                "phiBar": lambda simulation: self.expectation(simulation, None),  # Default expectation
                "energy": lambda simulation: simulation.action.findAction(simulation),
                "empty": lambda: 0,
            }

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

        return M / lattice.Ntot
    
    def computeObservable(self, simulation, name):
        lattice = simulation.lattice
        workingLattice = simulation.workingLattice

        try:
            func = self.observables[name]
        except KeyError:
            raise ValueError(f"Unknown observable: {name}")
        return func(simulation)


    def recordObservable(self, simulation, value=None):
        if value is None:
            if self.observableFuncName is not None:
                try:
                    value = self.computeObservable(simulation, self.observableFuncName)
                except AttributeError:
                    raise AttributeError(f"Action class must implement 'computeObservable(name)' method.")
            else:
                value = 0  # Or skip recording
        
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


