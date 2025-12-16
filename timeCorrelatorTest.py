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




def twoPointCorr(self):
    
  C = 0
  CArray = np.zeros(self.Ntot**2)
  number = 0

  for n1 in range(self.Ntot):
    for n2 in range(self.Ntot):
      CArray[number] = self.lat[n1]*self.lat[n2]
      number += 1

  C = np.mean(CArray)
  CError = np.std(CArray)/(self.Ntot-1)
  return C, CError


def twoPointTimeCorr(simulation, configNumber, interconfigCycles):

  timesteps, spacesteps = simulation.lattice.latdims
        
  GCMatrix = np.zeros((timesteps+1,configNumber))

  for i in range(configNumber):
    #print(i)
    sim.updateCycles(interconfigCycles)

    for tau2 in range(timesteps+1):
      tau = tau2 % timesteps
      corr = 0.0

      for t2 in range(timesteps): # Here I don't replace the +1
        t = t2%timesteps
        for x in range(spacesteps):
          for y in range(spacesteps):
            n1 = t * spacesteps + x
            n2 = ((t + tau) % timesteps) * spacesteps + y
            corr += simulation.workingLattice[n1] * simulation.workingLattice[n2]

                

      corr /= (spacesteps * spacesteps * timesteps)
      GCMatrix[tau2,i] += corr

    print("Done config ",i, "/",configNumber)

  # Average over configurations
  GCArray = np.mean(GCMatrix, axis=1)
  GCErrors = np.std(GCMatrix, axis=1)/np.sqrt(configNumber-1)

  return GCArray, GCErrors



a = 2
exp1 = 2
exp2 = 2
exp3 = 2
pregameWarmCycles = int(10**exp1)
correlatorConfigs = int(10**exp2)#10**exp2)
interconfigCycles = int(10**exp3) # Each cycle is T*L updates


#(2,2),(4,4),(5,5),(8,8),(10,10),(12,12),(5,10),(14,8)
pairs = [(3,3)]
for pair in pairs:
  latdims = np.array(pair)
  T = latdims[0]
  L = latdims[1]

  #0.1,0.5,1.0,1.5
  for m in [1.0]:

    for proposerType in ["MH"]:
    
      action = Action(m=m)



      lattice = SquareND(latdims, shuffle=True)


      if proposerType == "HB":
        proposer = HeatbathProposer()
      elif proposerType == "MH":
        proposer = MetropolisProposer(dMax=1.0,distribution='gaussian')


      sim = Simulation(

        lattice=lattice,
        action=action,
        updateProposer=proposer,
        warmCycles=pregameWarmCycles
        )


      #for i in range(configNumber):
      #    print(str(i+1)+"/"+str(configNumber))
      #    sim.updateCycles(interconfigCycles)
      #    sim.ReaderWriter.writeConfig(sim,filename= "TestingData.bin")

      GCArray, GCErrors = twoPointTimeCorr(sim,correlatorConfigs,interconfigCycles)


      xAxis = np.arange(latdims[0]+1)


      analyticResults = getAnalyticCorrelator(lattice,m)



      plt.errorbar(xAxis,GCArray,GCErrors,label="Monte Carlo")


      titleString = f"{proposerType} Uniform {T}x{L}, m={m}, a=[{exp1},{exp2},{exp3}]"

      plt.title(f"MH Uniform, {T}x{L} lattice \n {pregameWarmCycles} warming cycles, {correlatorConfigs} configs, {interconfigCycles} cycles between configs")
      plt.xlabel("Tau")
      plt.ylabel("G(tau)")


      plt.plot(analyticResults,linestyle="dashed",label="Analytical")


      plt.legend()

      #Output Points to file

      outputData = np.zeros((latdims[0]+1,4))
      for i in range(latdims[0]+1):
          outputData[i,0] = i
          outputData[i,1] = GCArray[i]
          outputData[i,2] = GCErrors[i]
          outputData[i,3] = analyticResults[i]



      saveString = proposerType+"uniform" + str(T)+"x"+str(L)+",m="+str(m)+",a="+str(exp1)+str(exp2)+str(exp3)

      #np.savetxt("figures/figureData/"+saveString+".txt",outputData,header="Tau, MC_G(tau), MC_Error, Analytic_G(tau)")
      #plt.savefig("figures/"+saveString+".png")

      plt.show()
      #plt.close()