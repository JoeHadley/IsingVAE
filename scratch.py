import numpy as np
import random as r
import math

import matplotlib.pyplot as plt
from UpdateProposer import ToyMVAEProposer
from Lattice import SquareND
from Simulation import Simulation
from Action import Action
from Observer import Observer


def test_toymvae_site_distribution(proposer, lattice_size=10, cycles=100000, beta=1.0, m=1.0, dim=1):
    """
    Test ToyMVAEProposer per-site distribution vs theoretical Gaussian.
    """

    # Create a simple working lattice
    workingLattice = np.zeros(lattice_size)

    # Function to shift site with periodic boundary conditions
    def shift(site, i, direction):
        return (site + direction) % lattice_size

    # Store sampled values for a single site
    site_index = 0
    samples = []

    for _ in range(cycles):
        # Compute neighbor sum
        neighbourSum = 0
        for d in range(dim):
            neighbourSum += workingLattice[shift(site_index, d, 1)]
            neighbourSum += workingLattice[shift(site_index, d, -1)]

        # Conditional mean and stddev (same as in ToyMVAE)
        A = dim + 0.5*m**2
        meanHB = neighbourSum / (2*A)
        sigmaHB = math.sqrt(1/(2*beta*A))

        # Sample site
        z = r.gauss(0,1)
        workingLattice[site_index] = meanHB + sigmaHB*z

        samples.append(workingLattice[site_index])

    samples = np.array(samples)
    sample_mean = np.mean(samples)
    sample_var = np.var(samples)

    theoretical_var = 1/(2*beta*A)

    print("ToyMVAE per-site distribution test:")
    print(f"Sampled mean: {sample_mean:.5f}")
    print(f"Sampled variance: {sample_var:.5f}")
    print(f"Theoretical variance: {theoretical_var:.5f}")
    print(f"Variance ratio (sample/theory): {sample_var/theoretical_var:.3f}")

# Example usage
test_toymvae_site_distribution(None, lattice_size=3, cycles=100000, beta=1.0, m=1.0, dim=1)
