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
        


@ dataclass
class VAEProposer(UpdateProposer):
  
  input_dim: int
  latent_dim: int
  double_input: bool
  learning: bool
  device: str='cpu'
  VAEbeta: float = 1.0

  def __post_init__(self):


    # Nearest power of 2 to input_dim/2
    hidden_dim = int(2**(round(math.log2(self.input_dim/2))))




    self.VAE = VAE(self.input_dim, hidden_dim, self.latent_dim,self.double_input, self.device, self.VAEbeta, lr=1e-3)  # Example parameters
    
  def setLearning(self, learning):
    self.learning = learning

  def updateCycle(self, simulation,site=None):


    for i in range(self.batch_size):
      n = r.choice(self.addressList)
      self.update(simulation, site=n,learning=self.learning)


  def propose(self, simulation, site,learning=False):



  
    #print(f"VAEProposer update at site {site}, learning={learning}")  # Debug statement

    input_phi, window_dims = simulation.lattice.createWindow(site,self.input_dim)

    #make the input a tensor if it is not already
    if not isinstance(input_phi, torch.Tensor):
        input_phi = torch.tensor(input_phi, dtype=torch.float32)


    output_phi, log_alpha = self.VAE.runLoop(input_phi,learning)  # Run the VAE to get the proposed new field value

    new_lattice = simulation.lattice.insertWindow(simulation.workingLattice, output_phi.detach().numpy(), site, window_dims)

    old_action = simulation.action.findAction(simulation)
    new_action = simulation.action.findAction(simulation,overrideWorkingLattice=new_lattice)
    dS = new_action - old_action
    log_alpha += -  dS
    

    acceptance_prob = torch.exp(log_alpha).item()  # Convert log_alpha to a scalar acceptance probability

    #acceptance_prob = self.VAE.compute_acceptance_probability(input_phi, output_phi)  # Compute acceptance probability

    output_phi = output_phi.detach().detach().numpy()  # Convert output to numpy array
    acceptance_prob = torch.exp(log_alpha).item()  # Convert log_alpha to a scalar acceptance probability

    # Generate a random number to decide acceptance
    roll = r.uniform(0, 1)
    accepted = roll < acceptance_prob

    return output_phi, acceptance_prob




@ dataclass
class HeatbathProposer(UpdateProposer):


  def propose(self,simulation,site=None):
  
    m = simulation.action.m
    dim = simulation.lattice.dim
    D = 0.5*m**2 + dim
    N = simulation.action.sumNeighbours(simulation, site ) 

    # Correct mean and stddev for conditional Gaussian:
    mean = N / (2*D)
    stddev = math.sqrt(1.0 / (2 * D))

    # draw new value
    new_value = r.gauss(mean, stddev)
    old_value = simulation.workingLattice[site]
    new_lattice = simulation.workingLattice.copy()
    new_lattice[site] = new_value
    
    boltz_weight_old = math.exp(-simulation.action.findAction(simulation))
    boltz_weight_new = math.exp(-simulation.action.findAction(simulation, overrideWorkingLattice=new_lattice))

    transition_prob_forward = (1 / (stddev * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((new_value - mean) / stddev) ** 2)
    transition_prob_backward = (1 / (stddev * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((old_value - mean) / stddev) ** 2)

    acceptance_probability =  min(1.0, (boltz_weight_new * transition_prob_backward) / (boltz_weight_old * transition_prob_forward))
    print("Acceptance Probability (Heatbath): ", (boltz_weight_new * transition_prob_backward) / (boltz_weight_old * transition_prob_forward))
    return new_lattice, acceptance_probability


@ dataclass
class MetropolisProposer(UpdateProposer):
  dMax: float = 2.0
  distribution: str = 'uniform'  # 'uniform' or 'gaussian'

  def propose(self, simulation, site):

    if self.distribution == 'gaussian':
      d = r.gauss(0, self.dMax)
    elif self.distribution == 'uniform':
      d = r.uniform(-self.dMax, self.dMax)
    elif self.distribution == 'pareto':
      d = r.paretovariate(self.dMax)

    new_lattice = simulation.workingLattice.copy()
    new_lattice[site] += d
    
    
    dS = simulation.action.actionChange(simulation, site,d)

    acceptance_probability = min(1,np.exp(-dS))
    return new_lattice, acceptance_probability
        

        
@dataclass
class DummyProposer(UpdateProposer):
  dMax: float = 1.0

  def propose(self, simulation,site=None):
    latdims = simulation.lattice.latdims
    newLattice = np.zeros(latdims)
    acceptanceProbability = 1.0

    return newLattice, acceptanceProbability

@ dataclass
class ToyMVAEProposer(UpdateProposer):


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