
import torch
import torch.nn as nn
#import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset


from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device='cpu',beta=1.0,lr=1e-3):
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
        #TODO Investigate best stack of layers

        self.optimizer = optim.Adam(self.parameters(), lr=lr)


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


  

    def log_prob_q(self, z, phi):
        """
        Computes log q(z | phi), assuming q is a diagonal Gaussian.
        z: Tensor of shape (latent_dim,) or (batch_size, latent_dim)
        phi: input to encoder (same shape as input_dim)

        Returns:
            log_prob: log probability of z under q(z|phi)
        """

        #print(f"z: {z}, phi: {phi}")
        #print(f"z type: {type(z)}, phi type: {type(phi)}")

        z_mean, z_logvar = self.encode(phi)
        var = torch.exp(z_logvar)
        log2pi = torch.log(torch.tensor(2.0 * torch.pi))

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



    def compute_decoder_jacobian(self, input_z):
        """
        Compute the Jacobian of decoder(z) ∈ ℝ³ with respect to z ∈ ℝ²
        """
        z = input_z.clone().detach().requires_grad_(True)
        output = self.decode(z)  # output shape: (3,)
        
        jacobian = []
        for i in range(output.shape[0]):
            grad_i = torch.autograd.grad(output[i], z, retain_graph=True, create_graph=True)[0]
            jacobian.append(grad_i)
            
        jacobian = torch.stack(jacobian, dim=0)  # shape: (3, 2)
        return jacobian

    def compute_jacobian_term(self, input_z):
        jacobian = self.compute_decoder_jacobian(input_z)
        JTJ = jacobian.T @ jacobian  # Compute Jacobian transpose @ Jacobian
        detTerm = torch.sqrt(torch.linalg.det(JTJ))
        return detTerm


    def compute_acceptance_probability(self, input_phi, output_phi):
        """
        Compute the acceptance probability for the proposed update.
        :param input_phi: Current field value
        :param output_phi: Proposed new field value
        :return: Acceptance probability
        """
        

        log_q_zF = self.log_prob_q(output_phi, input_phi)
        log_q_zB = self.log_prob_q(input_phi, output_phi)
        log_det_jF = torch.log(self.compute_jacobian_term(output_phi))
        log_det_jB = torch.log(self.compute_jacobian_term(input_phi))
        log_alpha = ( log_q_zF - log_q_zB - log_det_jF + log_det_jB
        )

        acceptance_prob = torch.exp(log_alpha).clamp(max=1.0)
        return acceptance_prob.item()  # Return as a scalar value

    def forward(self,phi):
        mean, logvar = self.encode(phi)
        z = self.reparameterization(mean, logvar)
        return self.decode(z)

    
    def compute_log_alpha(self, input_phi, output_phi, zF, zB):
        """
        Compute the log acceptance ratio for the proposed update.
        :param input_phi: Current field value
        :param output_phi: Proposed new field value
        :return: Log acceptance ratio
        """
        # Compute proposal densities
        log_q_zF = self.log_prob_q(zF, input_phi)
        log_q_zB = self.log_prob_q(zB, output_phi)
        # Jacobian magnitude (can use norm of df/dz or autodiff)
        log_det_jF = torch.log(self.compute_jacobian_term(zF))
        log_det_jB = torch.log(self.compute_jacobian_term(zB))

        # Log probabilities of initial and final states
        #log_p_phi = self.log_p(phi, phi, mu0=0, sigma0=1, sigma_y=1)
        #log_p_phi_prime = self.log_p(phi2, phi2, mu0=0, sigma0=1, sigma_y=1)

        # Compute log of acceptance ratio (up to target densities)
        log_alpha = ( log_q_zF - log_q_zB
                     - log_det_jF + log_det_jB
        )
        return log_alpha

    
    def runLoop(self, phi): # phi is the input tensor
        """
        Run a forward pass through the VAE and train.
        :param phi: Input tensor
        :return: Reconstructed output, mean, and log variance
        """

        mean, logvar = self.encode(phi)

        zF = self.reparameterization(mean, logvar)
        phi2 = self.decode(zF)  # Reconstructed output

        zB_mean, _ = self.encode(phi2)
        zB = zB_mean
        phi3 = self.decode(zB)  # Reconstructed output from inverse z
        # Will remain to be seen if phi3 is near-equal to phi, may need to adjust loss to insist on this



        log_alpha = self.compute_log_alpha(phi, phi2, zF, zB)




        # MH Loss (maximize acceptance)
        mh_loss = log_alpha.clamp(max=0)

        # KL loss (like VAE)
        kl_loss = self.computeKLLoss(phi, phi2, mu0=0, sigma0=1, sigma_y=1, forward_model=self.decoder)


        # Total loss
        loss = mh_loss + self.beta * kl_loss
        loss.backward()


        self.optimizer.step()

        return phi2, log_alpha

myVAE = VAE(input_dim=3, hidden_dim=2, latent_dim=1, device='cpu', beta=1.0)
# Example usage

# Assuming you have some input tensor `phi`
phi = torch.randn(3)  # Example input tensor
myVAE.runLoop(phi)

