import numpy as np

class Observer:
    def __init__(self, observableFuncName=None, recordWhileWarming=False, historyLimit=1000):
        self.observableFuncName = observableFuncName
        self.recording = observableFuncName is not None
        self.recordWhileWarming = recordWhileWarming if self.recording else False

        self.historyLimit = historyLimit
        self.history = np.zeros(historyLimit)
        self.historyCount = 0
        self.historyLimitReached = False

    def recordObservable(self, Action, value=None):
        if value is None:
            if self.observableFuncName is not None:
                try:
                    value = Action.computeObservable(self.observableFuncName)
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


