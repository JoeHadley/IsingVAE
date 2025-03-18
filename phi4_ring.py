import numpy as np
import random as r
import matplotlib.pyplot as plt
from analyticSolution import *
from Lattice import *
import time

def getCoords(n, sideLength):
    row = int(n//sideLength)
    col = int(n%sideLength)
    return row,col

def plot(data):
    plt.plot(data)

    plt.show()
            
# test change


T = 10
L = 10
latdims = np.array((T,L))

a = 3

exp1 = a
exp2 = a
exp3 = a
pregameWarmCycles = int(10**exp1)
correlatorConfigs = int(1)#10**exp2)
interconfigCycles = int(10**exp3) # Each cycle is T*L updates


lat = Lattice(latdims, warmCycles= pregameWarmCycles,dMax0=.9)

#lat.metroCycles(interconfigCycles)
#lat.writeConfig("output.bin"

configNumber = 1000
for i in range(configNumber):
    print(str(i+1)+"/"+str(configNumber))
    lat.metroCycles(interconfigCycles)
    lat.writeConfig("TestingData.bin")







#GCArray, GCErrors = lat.twoPointTimeCorr(correlatorConfigs,interconfigCycles)
#constant = GCArray[0]


#folder = "Results\\10x10\\"

#filePath = folder + "GCArray"+"10^"+str(exp1)+"10^"+str(exp2)+ "10^"+str(exp3)+".txt"
#np.savetxt(filePath, GCArray)

#filePath = folder + "GCErrors"+"10^"+str(exp1)+"10^"+str(exp2)+ "10^"+str(exp3)+".txt"
#np.savetxt(filePath, GCErrors)




# Analytic calculation

#analyticResults = getAnalyticCorrelator(lat)


#print(analyticResults)





#xAxis = np.arange(latdims[0]+1)
#C, CError = lat.twoPointCorr()

#plt.errorbar(xAxis,GCArray,GCErrors,label="Monte Carlo")
#plt.title(f"{T}x{L} lattice, {pregameWarmCycles} warming cycles, {correlatorConfigs} configs, {interconfigCycles} cycles between configs")
#plt.xlabel("Tau")





#plt.plot(analyticResults,linestyle="dashed",label="Analytical")


#plt.legend()

#plt.show()