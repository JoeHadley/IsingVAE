import torch
import torch.nn as nn
#import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset


from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid




def f(z):
    # z is shape (2,), output is also shape (2,)
    return torch.stack([
        z[0] ** 2 + z[1],   # f1 = x₀² + x₁
        z[0] * z[1]         # f2 = x₀·x₁
    ])

z = torch.tensor([1.0, 2.0])
output = f(z)

def jacobian_norm(z, f):
    z = z.clone().detach().requires_grad_(True)
    output = f(z)  # tensor of shape (2,)
    grad = torch.autograd.grad(outputs=output, inputs=z,
                                grad_outputs=torch.ones_like(output),
                                create_graph=True)[0]
    return grad.norm()


print(jacobian_norm(z, f))