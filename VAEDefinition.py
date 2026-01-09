
import torch
import torch.nn as nn
#import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset


from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid


class VAE(nn.Module):
    def __init__(self, window_dim, hidden_dim, latent_dim, double_input = False, device='cpu',beta=1.0,lr=1e-3):
        super(VAE, self).__init__()
        self.device = device   
        self.beta = beta
        self.window_dim = window_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.double_input = double_input
        

        

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(window_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )
        

        # Latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, latent_dim)
        self.logvar_layer = nn.Linear(latent_dim, latent_dim)

        
        
        # Decoder - Option to modify
        if self.double_input:

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim + window_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, window_dim)
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, window_dim)
            )
        # Output sigmoid removed

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

    def decode(self, z, input_phi=None):



        if self.double_input:
            # Concatenate z with original input
            decoder_input = torch.cat((z, self.input_phi ), dim=-1)
        else:
            decoder_input = z
        return self.decoder(decoder_input)

  

    def log_prob_q(self, z, phi):
        """
        Computes log q(z | phi), assuming q is a diagonal Gaussian.
        z: Tensor of shape (latent_dim,) or (batch_size, latent_dim)
        phi: input to encoder (same shape as window_dim)

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
        mean, logvar = self.mean, self.logvar

        # Compute log likelihood and prior
        #log_likelihood_value = self.log_likelihood(phi, data, sigma_y, forward_model)
        #log_prior_value = self.log_prior(phi, mu0, sigma0)

        # Compute KL divergence loss
        kl_loss = -0.5 *self.beta* torch.sum(1 + logvar - mean.pow(2) - logvar.exp())



        return kl_loss





    def compute_decoder_jacobian(self, input_z):
        """
        Compute the Jacobian of decoder(z) ∈ ℝ³ with respect to z ∈ ℝ²
        """
        z = input_z.detach().requires_grad_(True)
        output = self.decode(z) 
        
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


    def compute_exponential_term(self, input_z, mean, logvar):
        # Want sum of squares of input_z weighted by variance
        var = torch.exp(logvar)              # variance
        diff = input_z - mean                # z - mean
        exp_term = 0.5 * torch.sum(diff**2 / var)
        return torch.exp(exp_term)
    

    def compute_log_alpha(self, input_phi, output_phi, zF, zB, mean, logvar, back_mean, back_logvar):
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



      # Compute exponential factors
      log_zSumF = torch.log(self.compute_exponential_term(zF, mean, logvar))
      log_zSumB = torch.log(self.compute_exponential_term(zB, back_mean, back_logvar))

      # Compute Sigma ratio factors
      #Sigma_ratioF = 0.5*torch.log(torch.sum(torch.exp(logvar)))
      #Sigma_ratioB = 0.5*torch.log(torch.sum(torch.exp(back_logvar)))

        # logvar is a tensor of log variances. I want the norm of them
        # Variance means sigma^2, so exp(logvar) gives variance

      Sigma_ratioOld = 0.5 * torch.sum(logvar)
      Sigma_ratioNew = 0.5 * torch.sum(back_logvar)



      # Compute log of acceptance ratio (up to target densities)
      log_alpha = (- log_det_jF + log_det_jB + log_zSumB - log_zSumF + Sigma_ratioNew - Sigma_ratioOld
      )
      return log_alpha

  
    def runLoop(self, phi, learning): # phi is the input tensor


        self.input_phi = phi  # Store input for decoder if double_input is used 
        mean, logvar = self.encode(phi)

        self.mean=mean
        self.logvar=logvar



        zF = self.reparameterization(mean, logvar)
        phi2 = self.decode(zF, phi)

        back_mean, back_logvar  = self.encode(phi2)
        #self.back_mean=back_mean
        #self.back_logvar=back_logvar
        zB = self.reparameterization(back_mean, back_logvar)

        phi3 = self.decode(zB, phi2)




        log_alpha = self.compute_log_alpha( phi, phi2, zF, zB, mean, logvar, back_mean, back_logvar)

        # acceptance Loss (maximize acceptance)
        acc_loss = log_alpha.clamp(max=0)
        # Recon Loss (MSE)
        # KL loss (like VAE)
        kl_loss = self.computeKLLoss(phi, phi2, mu0=0, sigma0=1, sigma_y=1, forward_model=self.decoder)
        # Total loss
        loss = acc_loss + self.beta * kl_loss

        if learning:
            



            loss.backward()
            self.optimizer.step()



        return phi2, log_alpha

#myVAE = VAE(window_dim=3, hidden_dim=2, latent_dim=1, device='cpu', beta=1.0)
# Example usage

# Assuming you have some input tensor `phi`
#phi = torch.randn(3)  # Example input tensor
#myVAE.runLoop(phi, learning=False)

