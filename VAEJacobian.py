from VAEDefinition import VAE
import torch



    super(VAE, self).__init__()
    self.device = device   
    self.beta = beta
    self.window_dim = window_dim
    self.hidden_dim = hidden_dim
    self.latent_dim = latent_dim
    self.double_input = double_input
      


myVAE = VAE(window_dim=16, hidden_dim=8, latent_dim=2, double_input=True, device='cpu', VAEbeta=1.0)

