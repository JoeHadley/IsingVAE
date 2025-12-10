import numpy as np
import os
import matplotlib.pyplot as plt
from Simulation import Simulation
from Lattice import Lattice, SquareND
from UpdateProposer import *
from Observer import *
from Action import *


Nx=4
Ny=2
Nz=0

latdims = [Nx,Ny]


myAction = Action(m=2.0)
myLattice = SquareND(latdims)
#myUpdateProposer = VAEProposer(input_dim=4, latent_dim=2, double_input=False, learning=True, device='cpu', MCbeta=1.0, VAEbeta=1.0)


distribution = "gaussian"  # Options: "uniform", "gaussian"

myUpdateProposer = MetropolisProposer(distribution=distribution)
myObserver = Observer("StructFactor",latdims=latdims)
mySimulation = Simulation(1.0, myLattice, myAction, myUpdateProposer, myObserver)

mySimulation.updateCycles(2000)

data = mySimulation.observer.history[:mySimulation.observer.historyCount]




average = np.mean(data, axis=0)
stddev = np.std(data, axis=0)



print("Average Structure Factor:")
print(average)
print("Standard Deviation:")
print(stddev)


# Theoretical prediction for comparison

kx = 2*np.pi*np.fft.fftfreq(Nx)
ky = 2*np.pi*np.fft.fftfreq(Ny)

kx2d, ky2d = np.meshgrid(kx, ky, indexing = 'ij')
khat2 = 4*np.sin(kx2d/2)**2 + 4*np.sin(ky2d/2)**2

m = 2.0

S_theory = 1/(m**2 + khat2)


print("Theoretical Structure Factor:")
print(S_theory)


#k_hat_squared = 4


