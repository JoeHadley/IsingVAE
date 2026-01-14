from VAEDefinition import VAE
import torch







myVAE = VAE(window_dim=16, hidden_dim=8, latent_dim=2, double_input=False, device='cpu', beta=1.0)


input_phi = torch.randn(1, 16)
print("Input phi:", input_phi)



mean, logvar = myVAE.encode(input_phi)
print("Encoded mean:", mean)
print("logvar:", logvar)
z_reparam = myVAE.reparameterization(mean, logvar)
print("Reparameterized z:", z_reparam)

output_phi = myVAE.decode(z_reparam, input_phi=input_phi)

backMean, backLogvar = myVAE.encode(output_phi)
print("Decoded output phi:", output_phi)
print("Re-encoded mean from output phi:", backMean)
print("Re-encoded logvar from output phi:", backLogvar)


