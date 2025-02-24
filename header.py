## Import everything needed

import numpy as np
import random as r

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import torch
import torch.nn as nn
#import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset


from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import base64


## Utility functions and Wolff update

def getCoords(address,sideLength):
    row = int(address//sideLength)
    col = int(address%sideLength)
    return row, col

def getAddress(row,col,sideLength):
    address = row*sideLength + col
    return address

def query(tensor,address):
    sideLength = tensor.size(dim=1)
    row,col = getCoords(address,sideLength)
    return tensor[row][col]

def getNeighbours(site,N,dim):

    size = 2*dim
    ds = np.zeros(size,dtype=int)
    neighbours = np.zeros(size,dtype=int)

    for i in range(size):
      d = np.ceil(0.5*(i+1))

      neighbours[i] =  int(N**d) * (site // int(N**d))          + ( site + int( (-1)**i ) * int(N**(d-1)) + int(N**dim) ) % int(N**d)
    return neighbours

def getEnergy(tensor):
    L = tensor.size(dim=1)
    disagrees = 0
    for row in range(L):
        for col in range(L):
            if tensor[row, col] != tensor[(row + 1) % L, col]:
                disagrees += 1
            if tensor[row, col] != tensor[row, (col + 1) % L]:
                disagrees += 1
    return disagrees

def scramble(tensor,includeNegative = False):
  L = tensor.size(dim=1)
  for row in range(L):
    for col in range(L):
      if includeNegative:
        tensor[row,col] = r.randint(0,1)*2 - 1
      else:
         tensor[row,col] = r.randint(0,1)
  return tensor

def tensorCut(largeTensor,size, patchLocation = [0,0]):
    # Get dimensions of both tensors
    largeLength = largeTensor.size()

    # Create smaller tensor
    smallTensor = torch.empty(size, size, dtype=largeTensor.dtype)

    for row in range(size):
       for col in range(size):
        rowOnLargeTensor = (patchLocation[0] + row)%largeLength[0]
        colOnLargeTensor = (patchLocation[1] + col)%largeLength[1]
        smallTensor[row,col] = largeTensor[rowOnLargeTensor,colOnLargeTensor]

    return smallTensor



def tensorPaste(largeTensor, smallTensor, patchLocation):
    # Get dimensions of both tensors
    smallLength = smallTensor.size()
    largeLength = largeTensor.size()

    # Calculate the starting and ending indices
    row_indices = (torch.arange(smallLength[0]) + patchLocation[0]) % largeLength[0]
    col_indices = (torch.arange(smallLength[1]) + patchLocation[1]) % largeLength[1]

    # Use meshgrid to create all combinations of row and column indices
    row_grid, col_grid = torch.meshgrid(row_indices, col_indices, indexing='ij')

    # Insert smallTensor into largeTensor using advanced indexing
    largeTensor[row_grid, col_grid] = smallTensor

    return largeTensor

def tensorRound(tensor):
  length = tensor.size()

  #tensorMax = tensor.max()

  for row in range(length[0]):
    for col in range(length[1]):
      #tensor[row,col] = tensor/tensorMax

      if tensor[row,col] > 0.5:
        tensor[row,col] = 1
      else:
        tensor[row,col] = 0

  return tensor


def wolffUpdate(tensor,temperature,verbose=False):

    L = tensor.size(dim=1)
    dim = tensor.dim()
    initialSite = r.randint(0,(L**2) -1)
    spin = query(tensor,initialSite)

    if verbose:
      print("site = ", initialSite)




    Fold = {initialSite}
    cluster = {initialSite}

    while Fold:
        Fnew = set()

        for site in Fold:
            neighbours = getNeighbours(site, L, dim)
            if verbose:
              print(neighbours)
            for neighbour in neighbours:
              if verbose:
                print("Queried ", neighbour)
              if query(tensor, neighbour) == spin:
                if verbose:
                  print("Spin agreed")
                if neighbour not in cluster:
                  if r.uniform(0, 1) < (1 - np.exp(-2 * 1 / temperature)):
                    if verbose:
                      print("Added ", neighbour, " to cluster")
                    Fnew.add(neighbour)
                    cluster.add(neighbour)
        Fold = Fnew

    if verbose:
      print("Cluster = ", cluster)
    for site in cluster:
        row, col = getCoords(site, L)
        tensor[row, col] *= -1

    return tensor



## NN Functions

#Load and preprocess data
def load_and_preprocess_data(data_path, labels_path, temps_path, side_length):
    with open(data_path, 'rb') as data_file:
        data = np.fromfile(data_file, dtype=np.int32)
    
    num_rows = int(data.size/(side_length**2))
    
    data = data.reshape(num_rows, side_length, side_length)


    with open(labels_path, 'rb') as label_file:
        labels = np.fromfile(label_file, dtype=np.int32)
    labels = labels.reshape(num_rows, 1)

    with open(temps_path, 'rb') as temps_file:
        temps = np.fromfile(temps_file, dtype=np.int32)
    temps = temps.reshape(num_rows, 1)

    return data, labels, temps


def processDataPhi4(data_path, side_length):
    

    num_lines = 0  # Counter for total lines

    with open(data_path, "r") as file:
        for line in file:
            num_lines += 1  # Count lines


    configs = np.zeros(shape=(num_lines,side_length**2))
    with open(data_path, "r") as file:
        for i in range(num_lines):
            binary_data = base64.b64decode(line.strip())
            configs[i,:] = np.frombuffer(binary_data, dtype=np.float64)


    return configs


def shuffle_data(data, labels, temps):
    num_rows = len(labels)
    indices = np.random.permutation(num_rows)
    data = data[indices]
    labels = labels[indices]
    temps = temps[indices]

    return data, labels, temps














class VAE(nn.Module):
  def __init__(self, input_dim=100, hidden_dim=50, latent_dim=2, device='cpu'):
    super(VAE, self).__init__()
    self.device = device

    # Encoder
    self.encoder = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.LeakyReLU(0.2),
      nn.Linear(hidden_dim, latent_dim),
      nn.LeakyReLU(0.2)
    )

    # Latent mean and variance
    self.mean_layer = nn.Linear(latent_dim, 2)
    self.logvar_layer = nn.Linear(latent_dim, 2)

    # Decoder
    self.decoder = nn.Sequential(
      nn.Linear(2, latent_dim),
      nn.LeakyReLU(0.2),
      nn.Linear(latent_dim, hidden_dim),
      nn.LeakyReLU(0.2),
      nn.Linear(hidden_dim, input_dim),
      nn.Sigmoid()
    )

  def encode(self, x):
    x = self.encoder(x)
    mean, logvar = self.mean_layer(x), self.logvar_layer(x)
    return mean, logvar

  def reparameterization(self, mean, logvar):
    std = torch.exp(0.5 * logvar)  # Calculate the standard deviation
    epsilon = torch.randn_like(std).to(self.device)  # Sample epsilon
    z = mean + std * epsilon  # Reparameterization trick
    return z

  def decode(self, z):
    return self.decoder(z)

  def forward(self, x):
    mean, logvar = self.encode(x)
    z = self.reparameterization(mean, logvar)
    x_hat = self.decode(z)
    return x_hat, mean, logvar  # Ensure consistent naming
  
  def generate_from_image(self, image):
    """
    This function takes an input image, encodes it to latent space, 
    and then decodes it to generate a new image.
    
    Args:
      image (torch.Tensor): The input image tensor.

    Returns:
      torch.Tensor: The generated image.
    """
    # Step 1: Encode the input image to get latent mean and logvar
    mean, logvar = self.encode(image)

    # Step 2: Sample from the latent distribution (reparameterization trick)
    z = self.reparameterization(mean, logvar)

    # Step 3: Decode the latent vector back into an image
    generated_image = self.decode(z)

    # Return the generated image and the latent space (mean) for reference
    return generated_image, mean
