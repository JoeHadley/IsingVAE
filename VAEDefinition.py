
import torch
import torch.nn as nn
#import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset


from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid


class VAE(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=50, latent_dim=2, device='cpu',beta=1.0):
        super(VAE, self).__init__()
        self.device = device   
        self.beta = beta
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )

        # Latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, latent_dim)
        self.logvar_layer = nn.Linear(latent_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        # Output sigmoid removed


    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)  # Calculate the standard deviation
        epsilon = torch.randn_like(std)  # Sample epsilon
        z = mean + std * epsilon  # Reparameterization trick
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
  

    def log_prob_q(self, z, phi):
        """
        Computes log q(z | phi), assuming q is a diagonal Gaussian.
        z: Tensor of shape (latent_dim,) or (batch_size, latent_dim)
        phi: input to encoder (same shape as input_dim)

        Returns:
            log_prob: log probability of z under q(z|phi)
        """
        z_mean, z_logvar = self.encode(phi)
        var = torch.exp(z_logvar)
        log2pi = torch.log(torch.tensor(2.0 * torch.pi, device=z.device))

        log_prob = -0.5 * ((z - z_mean) ** 2 / var + z_logvar + log2pi)
        return log_prob.sum(dim=-1)  # sum over latent dimensions

    #TODO: should f be any map from the latent space z to R^M
    def jacobian_norm(self,z, f):
        """
        Compute norm of Jacobian df/dz where f: z -> R^2
        Returns scalar norm ||df/dz|| = sqrt((df_x/dz)^2 + (df_y/dz)^2)
        """
        z = z.clone().detach().requires_grad_(True)
        output = f(z)  # Should return tensor of shape (2,)
        grad = torch.autograd.grad(outputs=output, inputs=z,
                                grad_outputs=torch.ones_like(output),
                                create_graph=True)[0]  # shape: (1,)
        #TODO: understand how autograd.grad works and whether it is efficient/correct
        #TODO: can we understand how it knows about the decoder?
        return grad.norm()

    

    def computeKLLoss(self, phi, data, mu0, sigma0, sigma_y, forward_model):
        """
        Compute the loss for the VAE.
        :param phi: Input tensor
        :param data: Observed data
        :param mu0: Prior mean
        :param sigma0: Prior standard deviation
        :param sigma_y: Observation noise standard deviation
        :param forward_model: Function to compute model predictions
        :return: Loss value
        """
        # Forward pass through the VAE
        mean, logvar = self.encode(phi)

        # Compute log likelihood and prior
        #log_likelihood_value = self.log_likelihood(phi, data, sigma_y, forward_model)
        #log_prior_value = self.log_prior(phi, mu0, sigma0)

        # Compute KL divergence loss
        kl_loss = -0.5 *self.beta* torch.sum(1 + logvar - mean.pow(2) - logvar.exp())



        return kl_loss

    def log_prior(self,phi, mu0, sigma0):
            """Log Gaussian prior: N(mu0, sigma0^2 I)"""
            d = phi.shape[0]
            var0 = sigma0**2
            diff = phi - mu0
            log_term = torch.log(torch.tensor(2.0 * torch.pi * var0, device=diff.device))
            logp = -0.5 * (d * log_term + (diff ** 2).sum() / var0)
            return logp

    def log_likelihood(self,phi, data, sigma_y):
        """
        Gaussian likelihood: p(data | phi) = N(data | g(phi), sigma_y^2 I)
        - forward_model(phi): predicts mean of data given phi
        """
        
        pred = self.forward(phi)  # Forward pass to get predictions

        
        var_y = sigma_y**2
        diff = data - pred
        d = data.shape[0]
        log_term = torch.log(torch.tensor(2.0 * torch.pi * var_y, device=diff.device))
        logl = -0.5 * (d * log_term + (diff ** 2).sum() / var_y)
        return logl

    def log_p(self,phi, data, mu0, sigma0, sigma_y):
        """Log posterior up to constant"""
        log_p = self.log_likelihood(phi, data, sigma_y) + self.log_prior(phi, mu0, sigma0)
        return log_p

    def forward(self,phi):
        mean, logvar = self.encode(phi)
        z = self.reparameterization(mean, logvar)
        return self.decode(z)

    def runLoop(self, phi,optimiser): # phi is the input tensor
        """
        Run a forward pass through the VAE and train.
        :param phi: Input tensor
        :return: Reconstructed output, mean, and log variance
        """

        mean, logvar = self.encode(phi)
        z = self.reparameterization(mean, logvar)

        delta_phi = self.decode(z)
        phi_prime = phi + delta_phi




        # Inverse latent (solve z' such that decoder(z') = phi - phi_prime)
        #z_inv = self.encoder(phi_prime) #TODO: We know that z transforms phi to phi_prime via the decoder. 
        #TODO: here, we need to find z' such that phi = decoder(z') + phi_prime, this is the inverse transformation from phi_prime to phi.

        z_inv_mean, _ = self.encode(phi_prime)
        z_inv = z_inv_mean


        # Compute proposal densities
        log_q_z = self.log_prob_q(z, phi)
        log_q_z_inv = self.log_prob_q(z_inv, phi_prime)

        # Jacobian magnitude (can use norm of df/dz or autodiff)
        log_det_j = torch.log(self.jacobian_norm(z, self.decode))
        log_det_j_inv = torch.log(self.jacobian_norm(z_inv, self.decode))

        log_p_phi = self.log_p(phi, phi, mu0=0, sigma0=1, sigma_y=1)
        log_p_phi_prime = self.log_p(phi_prime, phi_prime, mu0=0, sigma0=1, sigma_y=1)

        # Compute log of acceptance ratio (up to target densities)
        log_alpha = (
            log_p_phi_prime - log_p_phi
            + log_q_z_inv - log_q_z
            + log_det_j - log_det_j_inv
        )






        # MH Loss (maximize acceptance)
        mh_loss = -log_alpha.clamp(max=0)

        # KL loss (like VAE)
        kl_loss = self.computeKLLoss(phi, phi_prime, mu0=0, sigma0=1, sigma_y=1, forward_model=self.decoder)

        # Total loss
        loss = mh_loss + self.beta * kl_loss
        loss.backward()
        optimiser.step()


myVAE = VAE(input_dim=100, hidden_dim=50, latent_dim=2, device='cpu', beta=1.0)
# Example usage
optimizer = optim.Adam(myVAE.parameters(), lr=1e-3)
# Assuming you have some input tensor `phi`
phi = torch.randn(100)  # Example input tensor
myVAE.runLoop(phi, optimizer)
