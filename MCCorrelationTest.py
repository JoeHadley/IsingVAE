import numpy as np
import random as r
import matplotlib.pyplot as plt
from analyticSolution import *
from Lattice import Lattice, SquareND
import time

from Lattice import Lattice, SquareND
from Action import Action
from Simulation import Simulation
from UpdateProposer import MetropolisProposer, HeatbathProposer
from ReaderWriter import ReaderWriter
import os
from Observer import Observer

def getCoords(n, sideLength):
    row = int(n//sideLength)
    col = int(n%sideLength)
    return row,col

def plot(data):
    plt.plot(data)

    plt.show()


def mcCorrelator(config1,config2):
  # Assume both configs are 1D arrays of length Ntot
  correlator = 0.0
  Ntot = len(config1)
  for n in range(Ntot):
    correlator +=  np.dot(config1,config2)
    
  correlator /= Ntot
  return correlator


import numpy as np

def connectedCorrelator(config1, config28):
    # Convert to numpy arrays
    a = config1
    b = config2
    
    # Means
    mu_a = a.mean()
    mu_b = b.mean()

    # Fluctuations
    da = a - mu_a
    db = b - mu_b

    # Variances
    var_a = np.mean(da * da)
    var_b = np.mean(db * db)

    # Handle edge cases: constant configurations
    if var_a <=1e-13 or var_b <=  1e-13:
        return np.nan

    # Connected covariance
    cov = np.mean(da * db)

    # Normalized connected correlator
    return cov / np.sqrt(var_a * var_b)



T = 10
L = 10
latdims = np.array((T,L))



exp1 = 2
exp2 = 2
exp3 = 0
pregameWarmCycles = int(10**exp1)
correlatorConfigs = int(10**exp2)#10**exp2)
interconfigCycles = int(10**exp3) # Each cycle is T*L updates


lattice = SquareND(latdims, shuffle=True)
m = 0.1
action = Action(m=m)

proposerType = "MH"

if proposerType == "HB":
  proposerTypeLong = "Heatbath"
  proposer = HeatbathProposer()
elif proposerType == "MH":
  proposerTypeLong = "Metropolis"
  proposer = MetropolisProposer(dMax=1.0)


sim = Simulation(
  lattice=lattice,
  action=action,
  updateProposer=proposer,
  warmCycles=pregameWarmCycles
  )




timesteps = correlatorConfigs
runsNumber = 100
correlatorArray = np.zeros((timesteps+1,runsNumber))

conCorrArray = np.zeros((timesteps+1,runsNumber))

for run in range(runsNumber):
  
  sim.updateCycles(pregameWarmCycles)
  referenceConfig = sim.workingLattice.copy()


  
  print("Run ",run+1,"/",runsNumber)

  correlatorArray[0,run] = mcCorrelator(referenceConfig,referenceConfig)
  conCorrArray[0,run] = connectedCorrelator(referenceConfig,referenceConfig)

  for time in range(timesteps):
    
    sim.updateCycles(interconfigCycles)

    # Calculate correlation between referenceConfig and current config
    
    correlatorArray[time+1,run] = mcCorrelator(referenceConfig,sim.workingLattice)
    conCorrArray[time+1,run] = connectedCorrelator(referenceConfig,sim.workingLattice)


# take first run

correlatorData = conCorrArray[:,0]

meanCorrelator = np.mean(conCorrArray,axis=1)
stdCorrelator = np.std(conCorrArray,axis=1)/np.sqrt(runsNumber-1)



xAxis = np.arange(timesteps+1)*interconfigCycles

for run in range(1,runsNumber):
  plt.plot(xAxis,conCorrArray[:,run],color='blue',alpha=0.01)
#plt.plot(xAxis,correlatorData,label="MC Correlator")

plt.plot(xAxis,meanCorrelator,color='black',linewidth=2,label="Mean Correlator")

print(conCorrArray)
plt.title(f"Connected Correlator over MC time, {proposerTypeLong} proposer \n {T}x{L} lattice, m={m}, 100 runs")
plt.xlabel("MC Cycles")
plt.ylabel("Connected Correlator")
plt.legend()
plt.show()