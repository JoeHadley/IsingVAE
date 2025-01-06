import numpy as np
import random as r
import matplotlib.pyplot as plt
from analyticSolution import *
from Lattice import *

def getCoords(n, sideLength):
    row = int(n//sideLength)
    col = int(n%sideLength)
    return row,col

            





##############################################################
##############################################################


T = 10
L = 10

pregameWarmCycles = 1000
correlatorConfigs = 10
interconfigCycles = 10



latdims = np.array((T,L))
lat = Lattice(latdims,warmCycles= pregameWarmCycles,dMax0=0.9)



lat.scrambleLattice()
lat.metroCycles(pregameWarmCycles)
print("Lattice warmed")


GCArray, GCErrors = lat.twoPointTimeCorr(correlatorConfigs,interconfigCycles)
constant = GCArray[0]









# Analytic calculation

analyticResults = getAnalyticCorrelator(lat)


print(analyticResults)





xAxis = np.arange(latdims[0]+1)
C, CError = lat.twoPointCorr()

plt.errorbar(xAxis,GCArray,GCErrors,label="Monte Carlo")
plt.title(f"Correlation function for {T}x{L} euclidean lattice between phi(x,0) and phi(y,tau)")
plt.xlabel("Tau")





plt.plot(analyticResults,linestyle="dashed",label="Analytical")


plt.legend()

plt.show()













