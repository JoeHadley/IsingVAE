import random as r
import math
import numpy as np
from abc import ABC, abstractmethod
from VAEDefinition import VAE
import torch
import base64



class UpdateProposer(ABC):



    @abstractmethod
    def update(self, simulation ,site = None):
        """
        Executes a single Monte Carlo update at `site` on the given `lattice`.
        Must update lattice.lat[site] in-place if accepted.
        Returns (accepted: bool, delta: float, dS: float)
        """
        pass



class VAEProposer(UpdateProposer):
    def __init__(self,input_dim, hidden_dim, latent_dim, device='cpu', beta=1.0):
        self.VAE = VAE(input_dim, hidden_dim, latent_dim, device, beta)  # Example parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim


def createWindow(simulation, site, window_size):
        # Same dimensions as wider lattice
    latdims = simulation.lattice.latdims
    dim = len(latdims)


    ntot = window_size**dim
    new_array = np.zeros(ntot)

    for i in range(ntot):
        number = np.base_repr(i, base=window_size).zfill(dim)
        moving_index = site
        for d in range(dim):
            
            digit = int(number[d])
            moving_index = simulation.lattice.shift(moving_index,d , digit)

        new_array[i] = simulation.workingLattice[moving_index]
        
    
    return new_array



    def updateCycle(self, simulation,site=None):
        pass

    def update(self, simulation, site=None):

        input_phi = simulation.lattice.lat  # Get the current field value at the site

        #make the input a tensor if it is not already
        if not isinstance(input_phi, torch.Tensor):
            input_phi = torch.tensor(input_phi, dtype=torch.float32)


        output_phi, log_alpha = self.VAE.runLoop(input_phi)  # Run the VAE to get the proposed new field value
        #acceptance_prob = self.VAE.compute_acceptance_probability(input_phi, output_phi)  # Compute acceptance probability

        output_phi = output_phi.detach().numpy()  # Convert output to numpy array

        acceptance_prob = torch.exp(log_alpha).item()  # Convert log_alpha to a scalar acceptance probability

        # Generate a random number to decide acceptance
        roll = r.uniform(0, 1)
        accepted = roll < acceptance_prob

        if accepted:
            simulation.workingLattice = output_phi  # Update the lattice with the new field value

        return accepted

class ModVAEProposer(UpdateProposer):
    def __init__(self,input_dim, hidden_dim, latent_dim, device='cpu', beta=1.0):
        self.VAE = VAE(input_dim, hidden_dim, latent_dim, device, beta)  # Example parameters

    def update(self, simulation, site=None):

        input_phi = simulation.lattice.lat  # Get the current field value at the site

        #make the input a tensor if it is not already
        if not isinstance(input_phi, torch.Tensor):
            input_phi = torch.tensor(input_phi, dtype=torch.float32)


        output_phi, log_alpha = self.VAE.runLoop(input_phi)  # Run the VAE to get the proposed new field value
        #acceptance_prob = self.VAE.compute_acceptance_probability(input_phi, output_phi)  # Compute acceptance probability

        output_phi = output_phi.detach().numpy()  # Convert output to numpy array

        acceptance_prob = torch.exp(log_alpha).item()  # Convert log_alpha to a scalar acceptance probability

        # Generate a random number to decide acceptance
        roll = r.uniform(0, 1)
        accepted = roll < acceptance_prob

        if accepted:
            simulation.workingLattice = output_phi  # Update the lattice with the new field value

        return accepted

# Mimics the MVAE structure using analytical f(z,phi) instead of a neural network
class ToyMVAEProposer(UpdateProposer):
  def __init__(self, beta=1.0,shuffle=False):
    self.beta = beta
    self.shuffle = True
    # Lazy initialization
    self.setupComplete = False
    self.Ntot = None
    self.addressList = None
  
  def shuffleList(self):
    self.addressList = np.random.permutation(self.addressList)


  def updateCycle(self, simulation):
    if not self.setupComplete:
      self.simulation = simulation
      self.Ntot = simulation.lattice.Ntot
      self.addressList = np.arange(self.Ntot)
      self.shuffleList()
      self.setupComplete = True
        
    if self.shuffle and self.addressList is not None:
      self.shuffleList()


    for i in range(self.Ntot):
      n = self.addressList[i]
      self.update(simulation, site=n)
  
  def update(self, simulation, site=None):

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


    output_phi = meanHB + sigmaHB*z
    # print(f"Site {site}: input_phi = {input_phi[site]}, output_phi = {output_phi}")
    # This should be exact, so acceptance prob = 1

    self.simulation.workingLattice[site] = output_phi

class HeatbathProposer(UpdateProposer):
    def __init__(self, beta=1.0,shuffle=False):
        self.beta = beta


        self.shuffle = shuffle

        # Lazy initialization
        self.setupComplete = False
        self.latdims = None
        self.Ntot = None
        self.addressList = None


    def shuffleList(self):
        self.addressList = np.random.permutation(self.addressList)

    def updateCycle(self, simulation):

        if not self.setupComplete:
            self.simulation = simulation
            self.Ntot = simulation.lattice.Ntot
            self.addressList = np.arange(self.Ntot)
            self.setupComplete = True


        if self.shuffle and self.addressList is not None:
            self.shuffleList()


        Ntot = simulation.lattice.Ntot
        for i in range(Ntot):
            n = self.addressList[i]
            self.update(simulation, site=n)

    def update(self, simulation, site=None,old=None):
        if not self.setupComplete:
            self.simulation = simulation
            self.Ntot = simulation.lattice.Ntot
            self.addressList = np.arange(self.Ntot)
            self.setupComplete = True

        n = site
        m = self.simulation.action.m
        dim = self.simulation.lattice.dim
        A = 0.5*m**2 + dim
        B= self.simulation.action.sumNeighbours(self.simulation, n )  # should be plain sum of neighbor phi

        # Correct mean and stddev for conditional Gaussian:
        mean = B / (2*A)
        stddev = math.sqrt(1.0 / (2*self.beta * A))

        # draw new value
        new_value = r.gauss(mean, stddev)

        # unconditional set for heatbath
        self.simulation.workingLattice[n] = new_value



    def update2(self, simulation, site = None,old=None):
        
        if not self.setupComplete:
            self.simulation = simulation
            self.Ntot = simulation.lattice.Ntot
            self.addressList = np.arange(self.Ntot)
            self.setupComplete = True

        
        
        n = site

        
        m = self.simulation.action.m
        dim = self.simulation.lattice.dim

        A = m**2 + 2*dim

        neighbourSum = self.simulation.action.sumNeighbours(self.simulation, n)

        B = neighbourSum * self.simulation.workingLattice[site]


        new_value = r.gauss(B/A, 1/A)  # Generate a new value from a Gaussian distribution
        
        self.simulation.workingLattice[n] = new_value


class MetropolisProposer(UpdateProposer):
    def __init__(self, dMax, beta=1.0, shuffle=False):
        self.dMax = dMax
        self.beta = beta
        self.shuffle = shuffle
        # Lazy initialization
        self.setupComplete = False
        self.Ntot = None
        self.addressList = None

    def shuffleList(self):
        self.addressList = np.random.permutation(self.addressList)

    def updateCycle(self, simulation):
        if not self.setupComplete:
            self.simulation = simulation
            self.Ntot = simulation.lattice.Ntot
            self.addressList = np.arange(self.Ntot)
            self.setupComplete = True
        
        if self.shuffle and self.addressList is not None:
            self.shuffleList()


        for i in range(self.Ntot):
            n = self.addressList[i]
            self.update(simulation, site=n)

    def printParams(self):
        print(f"Metropolis Proposer parameters: dMax = {self.dMax}, beta = {self.beta}")

    def update(self, simulation, site = None):



        n = site


        d = r.gauss(0,self.dMax)


        #dS = simAction.actionChange(simLattice, simulation.workingLattice, n,d)
        dS = simulation.action.actionChange2(simulation, n,d)

        boltFactor = np.exp(-dS*self.beta)

        p = min(1,boltFactor)

        roll = r.uniform(0,1)

        
        accepted = roll <= p
        
        if accepted:
            simulation.workingLattice[n] += d
        


        
        

        #if self.recording and (not self.warming or self.recordWhileWarming) and not self.historyLimitReached:

        #    self.recordObservable()






class DummyProposer(UpdateProposer):
    def __init__(self, dMax=1.0, beta=1.0):

        self.dMax = dMax
        self.beta = beta

    def updateCycle(self, simulation,site=None):
        self.update(simulation)

    def update(self, simulation, site=None):
        pass








