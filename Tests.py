
import numpy as np
import random as r
import matplotlib.pyplot as plt
from analyticSolution import *
from Lattice import *
import time
from header import processDataPhi4


#num_lines = sum(1 for _ in open('output.bin'))
#print(num_lines)

#lat = Lattice((10,10))
#lat.readConfig("output.bin",line_number=998)

#lat.show()

#showLat = np.reshape(lat.lat,lat.latdims)

#plt.imshow(showLat, cmap='coolwarm', origin='lower', aspect='equal')
#plt.colorbar(label="Value")
#plt.show()



data_path = "C:/Users/Joe/Documents/Projects/IsingVAE/TrainingData.bin"

configs = processDataPhi4(data_path,10)
print(configs)