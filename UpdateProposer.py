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


class heatbathProposer(UpdateProposer):
    def __init__(self, beta=1.0,shuffle=False):
        self.beta = beta

        m = 1

        # Lazy initialization
        self.setupComplete = False
        self.latdims = None
        self.Ntot = None
        self.addressList = None


    def shuffleList(self):
        self.addressList = np.random.permutation(self.addressList)

    def updateCycle(self, simulation):
        if self.shuffle and self.addressList is not None:
            self.shuffleList()


        for i in range(self.Ntot):
            n = self.addressList[i]
            self.update(simulation, site=n)

    def update(self, simulation, site = None):
        
        if not self.setupComplete:
            self.simulation = simulation
            self.Ntot = simulation.lattice.Ntot
            self.addressList = np.arange(self.Ntot)
            self.setupComplete = True

        
        
        n = site

        
        m = self.simulation.action.m
        dim = self.simulation.lattice.dim

        A = m**2 + 2*dim

        neighbourSum = self.simulation.action.neighborSum(self.simulation.lattice, self.simulation.workingLattice, n)

        B = neighbourSum * self.simulation.workingLattice[site]


        new_value = r.gauss(B/A, 1/A)  # Generate a new value from a Gaussian distribution
        
        self.simulation.workingLattice[n] = new_value


class MetropolisProposer(UpdateProposer):
    def __init__(self, dMax, beta=1.0):
        self.dMax = dMax
        self.beta = beta
        # Lazy initialization
        self.setupComplete = False
        self.Ntot = None
        self.addressList = None

    def shuffleList(self):
        self.addressList = np.random.permutation(self.addressList)

    def updateCycle(self, simulation):
        if self.shuffle and self.addressList is not None:
            self.shuffleList()


        for i in range(self.Ntot):
            n = self.addressList[i]
            self.update(simulation, site=n)

    def printParams(self):
        print(f"Metropolis Proposer parameters: dMax = {self.dMax}, beta = {self.beta}")

    def update(self, simulation, site = None):

        if not self.setupComplete:
            self.simulation = simulation
            self.Ntot = simulation.lattice.Ntot
            self.addressList = np.arange(self.Ntot)
            self.setupComplete = True



        n = site


        d = r.gauss(0,self.dMax)

        simAction = simulation.action
        simLattice = simulation.lattice

        #dS = simAction.actionChange(simLattice, simulation.workingLattice, n,d)
        dS = simAction.actionChange2(simLattice, simulation.workingLattice, n,d)

        boltFactor = np.exp(-dS*self.beta)

        p = min(1,boltFactor)

        roll = r.uniform(0,1)

        
        accepted = roll <= p
        
        if accepted:
            simulation.workingLattice[n] += d
        


        
        

        #if self.recording and (not self.warming or self.recordWhileWarming) and not self.historyLimitReached:

        #    self.recordObservable()











'''


    def metroUpdate(self,n):



        d = r.gauss(0,self.dMax)

        

        dS = self.actionChange(n,d)

        boltFactor = np.exp(-dS)

        p = min(1,boltFactor)

        roll = r.uniform(0,1)


        # Apply previous move to previousLat
        self.previousLat[int(self.previousMove[0])] += self.previousMove[1]

        # Apply this move to current lattice

        update = 0
        if roll <= p:
            update = d

        self.lat[n] += update
        self.previousMove[0] = n
        self.previousMove[1] = update # Is sometimes 0  
            #self.previousMove[0] = True


        
        

        if self.recording and (not self.warming or self.recordWhileWarming) and not self.historyLimitReached:

            self.recordObservable()

    def update(self, n):
        if self.proposer is None:
            raise ValueError("No update proposer is set.")
        
        accepted, d, dS = self.proposer.update(n, self)

        if self.recording and (not self.warming or self.recordWhileWarming) and not self.historyLimitReached:
            self.recordObservable()

        return accepted

    
    def metroCycle(self):
        if self.shuffle:
            self.shuffleList()


        for i in range(self.Ntot):
            n = self.addressList[i]
            self.metroUpdate(n)


    def updateCycle(self):
        self.proposer.updateCycle(self)

    def metroCycles(self,cycles):
        for c in range(cycles):
            self.metroCycle()

    def updateCycles(self,cycles):
        for c in range(cycles):
            self.updateCycle()

'''



