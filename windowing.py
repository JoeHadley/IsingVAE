import numpy as np

def insertWindow(L, largeLattice, l, smallLattice, site,dim=2):
  resultLattice = largeLattice.copy()
  if dim == 2:

    smallList = np.arange(0, l**dim)


    siteRow = site // L
    siteCol = site %  L
    siteCoord = np.array([siteRow, siteCol], dtype=int)

    for i in smallList:

      row = i // l
      col = i %  l
      smallCoord = np.array([row, col], dtype=int)
      newCoord = (smallCoord + siteCoord) % L
      
      newIndex = newCoord[0] * L + newCoord[1]

      resultLattice[newIndex] = smallLattice[i]
  
  elif dim == 1:
    for i in range(l):
      newIndex = (site + i) % L
      resultLattice[newIndex] = smallLattice[i]


  return resultLattice

def insertWindow2(self, large_lattice, window, site, window_dims):
  # Same dimensions as wider lattice
  dim = len(self.latdims)
  ntot = np.prod(window_dims)
  
  # Get coordinates of the site
  coords = np.zeros(dim, dtype=int)
  s = site
  for d in range(dim):
    coords[d] = (s // self.stride[d]) % self.latdims[d]
    s = site%self.stride[d]




  new_lattice = large_lattice.copy()

  for idx in range(ntot): 

    offset = idx
    new_coords = coords.copy()

    for d in reversed(range(dim)):
      step = offset % window_dims[d]
      
      offset //= window_dims[d]
      new_coords[d] = (coords[d] + step) % self.latdims[d]
    
    flat = 0
    for d in range(dim):
      flat += new_coords[d] * self.stride[d]
    new_lattice[flat] = window[idx]
    
    return new_lattice
  

