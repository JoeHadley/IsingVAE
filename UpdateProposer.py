import random as r
import math
import numpy as np
from abc import ABC, abstractmethod
from VAEDefinition import VAE
import torch
import base64
from dataclasses import dataclass



class UpdateProposer(ABC):



    @abstractmethod

    def propose(self, simulation,site=None):
      latdims = simulation.lattice.latdims
      newLattice = np.zeros(latdims)
      acceptanceProbability = 1

      return newLattice, acceptanceProbability
        



class VAEProposer(UpdateProposer):
  def __init__(self,lattice_dim, window_size, latent_dim, double_input =False,learning=False, batch_size=None, device='cpu', beta=1.0):

    self.window_size = window_size
    self.lattice_dim = lattice_dim
    self.batch_size = batch_size if batch_size is not None else self.Ntot
    self.input_dim = window_size**lattice_dim
    self.double_input = double_input
    self.learning = learning

    # Lazy initialization
    self.setupComplete = False
    self.Ntot = None
    self.addressList = None


    # Nearest power of 2 to input_dim/2
    hidden_dim = int(2**(round(math.log2(self.input_dim/2))))




    self.VAE = VAE(self.input_dim, hidden_dim, latent_dim,double_input, device, beta, lr=1e-3)  # Example parameters
    self.input_dim = self.input_dim
    self.hidden_dim = hidden_dim
    self.latent_dim = latent_dim

  def setLearning(self, learning):
    self.learning = learning

  def updateCycle(self, simulation,site=None):


    if not self.setupComplete:
      self.simulation = simulation
      self.Ntot = simulation.lattice.Ntot
      self.addressList = np.arange(self.Ntot)
      self.setupComplete = True  # Store learning flag



    for i in range(self.batch_size):
      n = r.choice(self.addressList)
      self.update(simulation, site=n,learning=self.learning)

  def update(self, simulation, site,learning=False):

    #print(f"VAEProposer update at site {site}, learning={learning}")  # Debug statement

    input_phi, window_dims = simulation.lattice.createWindow(site,self.window_size)

    #make the input a tensor if it is not already
    if not isinstance(input_phi, torch.Tensor):
        input_phi = torch.tensor(input_phi, dtype=torch.float32)


    output_phi, log_alpha = self.VAE.runLoop(input_phi,learning)  # Run the VAE to get the proposed new field value

    new_lattice = simulation.lattice.insertWindow(simulation.workingLattice, output_phi.detach().numpy(), site, window_dims)

    old_action = simulation.action.findAction(simulation)
    new_action = simulation.action.findAction(simulation,overrideWorkingLattice=new_lattice)
    dS = new_action - old_action
    log_alpha += -simulation.beta * dS
    

    acceptance_prob = torch.exp(log_alpha).item()  # Convert log_alpha to a scalar acceptance probability

    #acceptance_prob = self.VAE.compute_acceptance_probability(input_phi, output_phi)  # Compute acceptance probability

    output_phi = output_phi.detach().detach().numpy()  # Convert output to numpy array
    acceptance_prob = torch.exp(log_alpha).item()  # Convert log_alpha to a scalar acceptance probability

    # Generate a random number to decide acceptance
    roll = r.uniform(0, 1)
    accepted = roll < acceptance_prob

    if accepted:
        simulation.workingLattice = new_lattice  # Update the lattice with the new field value

    return accepted




@ dataclass
class HeatbathProposer(UpdateProposer):
  beta: float = 1.0

  def propose(self,simulation,site=None):
  
    m = simulation.action.m
    dim = simulation.lattice.dim
    A = 0.5*m**2 + dim
    B= simulation.action.sumNeighbours(simulation, site )  # should be plain sum of neighbor phi

    # Correct mean and stddev for conditional Gaussian:
    mean = B / (2*A)
    stddev = math.sqrt(1.0 / (2*self.beta * A))

    # draw new value
    new_value = r.gauss(mean, stddev)
    new_lattice = simulation.workingLattice.copy()
    new_lattice[site] = new_value
    
    acceptance_probability = 1.0

    return new_lattice, acceptance_probability





class MetropolisProposer(UpdateProposer):
  def __init__(self, dMax=2.0, beta=1.0):
    self.dMax = dMax
    self.beta = beta

  def propose(self, simulation, site):
    d = r.gauss(0,self.dMax)

    new_lattice = simulation.workingLattice.copy()
    new_lattice[site] += d
    
    
    dS = simulation.action.actionChange(simulation, site,d)

    acceptance_probability = np.exp(-dS*self.beta)
    return new_lattice, acceptance_probability
        

        
@dataclass
class DummyProposer(UpdateProposer):
  dMax: float = 1.0
  beta: float = 1.0

  def propose(self, simulation,site=None):
    latdims = simulation.lattice.latdims
    newLattice = np.zeros(latdims)
    acceptanceProbability = 1.0

    return newLattice, acceptanceProbability










# Mimics the MVAE structure using analytical f(z,phi) instead of a neural network
class ToyMVAEProposer(UpdateProposer):
  def __init__(self, beta=1.0):
    self.beta = beta

  def propose(self, simulation, site=None):

    # Set up parameters
    m = simulation.action.m
    dim = simulation.lattice.dim
    input_phi = simulation.workingLattice.copy()
    
    neighbourSum = 0
    for i in range(dim):
      neighbourSum += input_phi[simulation.lattice.shift(site, i, 1)]
      neighbourSum += input_phi[simulation.lattice.shift(site, i, -1)]
    
    meanHB =           (neighbourSum)/          (2*(dim+ m**2/2))
    sigmaHB =        (1             /(self.beta*2*(dim +m**2/2)))**0.5
    
    # Sample unit gaussian for z
    z = r.gauss(0,1)

    new_lattice = simulation.workingLattice.copy()
    new_lattice[site] = meanHB + sigmaHB*z

    acceptance_probability = 1.0 # This should be exact, so acceptance prob = 1

    return new_lattice, acceptance_probability