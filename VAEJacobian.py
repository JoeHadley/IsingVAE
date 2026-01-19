from VAEDefinition import VAE
import torch




window_dim=4
hidden_dim=2
latent_dim=1

myVAE = VAE(window_dim=window_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, double_input=False, device='cpu', beta=1.0)


input_phi = torch.randn(1, window_dim)
print("Input phi:", input_phi.tolist())



mean, logvar = myVAE.encode(input_phi)
print("mean:", mean.tolist())
print("logvar:", logvar.tolist())
zF = myVAE.reparameterization(mean, logvar)
print("Forward z:", zF.tolist())

output_phi = myVAE.decode(zF, input_phi=input_phi)

backMean, backLogvar = myVAE.encode(output_phi)

print("Decoded output phi:", output_phi)
print("Re-encoded mean from output phi:", backMean)
print("Re-encoded logvar from output phi:", backLogvar)

zB = myVAE.reparameterization(backMean, backLogvar)
print("Backward z:", zB)


def compute_decoder_jacobian( input_z):
    """
    Compute the Jacobian of decoder(z) ∈ ℝ³ with respect to z ∈ ℝ²
    """
    z = input_z.detach().requires_grad_(True)
    output = myVAE.decode(z).squeeze()

    jacobian = []
    for i in range(output.shape[0]):
        grad_i = torch.autograd.grad(
            output[i], 
            z, 
            retain_graph=True, 
            create_graph=True
        )[0]
        jacobian.append(grad_i)
    jacobian = torch.stack(jacobian, dim=0) # shape: (3, 2)
    return jacobian

def compute_jacobian_term(input_z, min_singular=1e-2):
    # Compute the decoder Jacobian
    J = compute_decoder_jacobian(input_z)  # shape (m, n)

    # Compute singular values
    _, S, _ = torch.linalg.svd(J, full_matrices=False)

    # Clamp singular values to a reasonable minimum
    S_clamped = torch.clamp(S, min=min_singular)

    # Log of product of singular values = log sqrt(det(J^T J))
    log_volume = torch.sum(torch.log(S_clamped))

    return log_volume


jacobian_term = compute_jacobian_term(zF)
print("Jacobian volume term (log sqrt(det(J^T J))):", jacobian_term.item())