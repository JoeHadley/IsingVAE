import numpy as np
import random as r
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid




m = 1
L = 10
T = 10

def comp_numerator(x1,x2,nx,t1,t2,nt):
    return np.exp(1j*2*np.pi*nx*(x2-x1)/L)*np.exp(1j*2*np.pi*nt*(t2-t1)/T)

def denominator(nx,nt,m=1):
    return 4*np.sin(np.pi*nx/L)**2 + 4*np.sin(np.pi*nt/T)**2 + m**2


def compAnCorr(x1,x2,t1,t2,m=1):
    result = 0

    for nx in range(L):
        for nt in range(T):
            #result2 =+ real_numerator(x1,x2,nx,t1,t2,nt)/denominator(nx,nt)
            result += comp_numerator(x1,x2,nx,t1,t2,nt)/denominator(nx,nt,m)
    return np.real_if_close(result/(L*T))



def getAnalyticCorrelator(lattice):
    T = lattice.latdims[0]
    L = lattice.latdims[1]
    m = lattice.m


    filePath = "AnalyticalCorrelator"+str(T)+"x"+str(L)+".txt"

    if os.path.exists(filePath):
        compResults = np.genfromtxt(filePath)
        print("Loaded from text file")
    else:

        compResults = np.zeros(T+1)

        for tau in range(T+1):

            for x in range(L):
                for y in range(L):
                    for t in range(T):
                        compResults[tau] += compAnCorr(x,y,t,(t+tau)%T,m)

        compResults /= (L*L*T)



        
        np.savetxt(filePath, compResults)

    return compResults
