import numpy as np
import random as r
import math
from Lattice import SquareND

latdims = np.array([3,3,3])

#for dim in range(1,5):

dim = 3

lat = SquareND(np.array(latdims))
lat.testMode()
lat.show()

print(lat.stride)
print(lat.vec)
window , window_dims = lat.createWindow(site=7, window_size=2)

show_window = np.reshape(window,window_dims)
print(show_window)
