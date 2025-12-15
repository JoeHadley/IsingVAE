import numpy as np
import random as r
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


# New simpler analytic correlator functions


def comp_numerator(t1,t2,nt,T):
    return np.exp(1j*2*np.pi*nt*(t2-t1)/T)

def denominator(nt,m,T):
    return 4*np.sin(np.pi*nt/T)**2 + m**2

def compAnCorr(t1,t2,m,T,L):
  result = 0

  for nt in range(T):
    #result2 =+ real_numerator(x1,x2,nx,t1,t2,nt)/denominator(nx,nt)
    result += comp_numerator(t1,t2,nt,T)/denominator(nt,m,T)
  return np.real_if_close(result/(L*T))

def getAnalyticCorrelator(lattice,m):
  T = lattice.latdims[0]
  L = lattice.latdims[1]



  compResults = np.zeros(T+1)

  for tau in range(T+1):
    for t in range(T):
      compResults[tau] += compAnCorr(t,(t+tau)%T,m,T,L)

  compResults /= T




  return compResults



def old_comp_numerator(x1,x2,nx,t1,t2,nt,T,L):
    return np.exp(1j*2*np.pi*nx*(x2-x1)/L)*np.exp(1j*2*np.pi*nt*(t2-t1)/T)

def old_denominator(nx,nt,m,T,L):
    return 4*np.sin(np.pi*nx/L)**2 + 4*np.sin(np.pi*nt/T)**2 + m**2

def old_compAnCorr(x1,x2,t1,t2,m,T,L):
    result = 0

    for nx in range(L):
        for nt in range(T):
            #result2 =+ real_numerator(x1,x2,nx,t1,t2,nt)/denominator(nx,nt)
            result += old_comp_numerator(x1,x2,nx,t1,t2,nt,T,L)/old_denominator(nx,nt,m,T,L)
    return np.real_if_close(result/(L*T))

def old_getAnalyticCorrelator(lattice,m):
    T = lattice.latdims[0]
    L = lattice.latdims[1]


    filePath = "analyticalCorrelators/AnalyticalCorrelator"+str(T)+"x"+str(L)+", m="+str(m)+".txt"

    if os.path.exists(filePath):
        compResults = np.genfromtxt(filePath)
        print("Loaded from text file")
    else:

        compResults = np.zeros(T+1)

        for tau in range(T+1):

            for x in range(L):
                for y in range(L):
                    for t in range(T):
                        compResults[tau] += old_compAnCorr(x,y,t,(t+tau)%T,m,T,L)

        compResults /= (L*L*T)



        np.savetxt(filePath, compResults)

    return compResults
