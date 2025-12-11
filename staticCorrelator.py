import numpy as np
import os
import matplotlib.pyplot as plt
from Simulation import Simulation
from Lattice import Lattice, SquareND
from UpdateProposer import *
from Observer import *
from Action import *


Nx=15
Ny=15
Nz=0

latdims = [Nx,Ny]

m = 1.0

myAction = Action(m=m)
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


S_theory = 1/(m**2 + khat2)


print("Theoretical Structure Factor:")
print(S_theory)


#k_hat_squared = 4

def radial_average(S2d, Nx, Ny, nbins = 20):
  kx = 2*np.pi*np.fft.fftfreq(Nx)
  ky = 2*np.pi*np.fft.fftfreq(Ny)
  kx2d, ky2d = np.meshgrid(kx, ky, indexing = 'ij')
  k_magnitude = np.sqrt(kx2d**2 + ky2d**2)

  kmax = np.max(k_magnitude)
  bin_edges = np.linspace(0, kmax, nbins+1)

  S_radial = []
  k_radial = []

  for i in range(nbins):
    mask = (k_magnitude >= bin_edges[i]) & (k_magnitude < bin_edges[i+1])
    if np.any(mask):
      S_radial.append(np.mean(S2d[mask]))
      k_radial.append(0.5*(bin_edges[i] + bin_edges[i+1]))

  return k_radial, S_radial



S_meas = average
S_th = S_theory
k_radial_meas, S_radial_meas = radial_average(S_meas, Nx, Ny, nbins=20)
k_radial_th, S_radial_th = radial_average(S_th, Nx, Ny, nbins=20)


print("Radially-Averaged Measured S(k):")
print(S_radial_meas)
print("Radially-Averaged Theoretical S(k):")
print(S_radial_th)

plt.figure(figsize=(6,4))
plt.plot(k_radial_meas, S_radial_meas, 'o-', label="Measured S(k)")
plt.plot(k_radial_th, S_radial_th, 's-', label="Theory S(k)")
plt.xlabel("|k|")
plt.ylabel("S(|k|)")
plt.title("Radially-Averaged Structure Factor, 15x15")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
