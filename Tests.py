
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









    def neighbTest(self,n=10,verbose=False):
        successes = 0
        for i in range(n):

            site = r.randint(0,self.Ntot-1)
            
            neighbs1 = self.getNeighbours(site)
            neighbs2 = self.getNeighbours2(site)

            miniSuccesses = 0
            if len(neighbs1) == len(neighbs2):

                for j in range(len(neighbs1)):
                    if neighbs1[j] == neighbs2[j]:
                        miniSuccesses +=1
            if miniSuccesses ==len(neighbs1):
                successes += 1 




            if verbose:
                print(f'Trial {i}: site: {site}, neighbours {neighbs1,neighbs2}')

        if successes == n:
            print(f"Neighb test passed for {n} trials")
        else:
            print("Neighb test failed")



    def neighbTest2(self,n=10,verbose=False):

        wins1 = 0
        wins2 = 0

        times1 = np.zeros(n)
        times2 = np.zeros(n)
        for i in range(n):


            sites = np.random.randint(0,self.Ntot-1,n)


            start = time.time()
            for j in range(n):
                neighbs1 = self.getNeighbours(sites[j])
            end = time.time()
            time1 = end-start
            times1[i] = time1

            
            start = time.time()
            for j in range(n):
                neighbs2 = self.getNeighbours(sites[j])
            end = time.time()
            time2 = end-start
            times2[i] = time2

            if time1 < time2:
                wins1 += 1
            elif time2< time1:
                wins2 += 1
        if wins1 > wins2:
            print("Method 1 wins", np.mean(times1), np.mean(times2))
        elif wins2 > wins1:
            print("Method 2 wins", np.mean(times1), np.mean(times2))
        else:
            print("A tie!", np.mean(times1), np.mean(times2))

def shiftTest(self,n=10,verbose=False):
        successes = 0
        for i in range(n):

            site = r.randint(0,self.Ntot-1)
            jump0 = r.randint(-2*self.latdims[0],2*self.latdims[0])
            jump1 = r.randint(-2*self.latdims[1],2*self.latdims[1])
            shiftedSite11 = self.shift(site, 0,jump0)
            shiftedSite12 = self.shift(shiftedSite11,1,jump1)

            shiftedSite21 = self.shift(site, 1, jump1)
            shiftedSite22 = self.shift(shiftedSite21, 0, jump0)

            if shiftedSite12 == shiftedSite22:
                successes += 1

            if verbose:
                print(f'Trial {i}: site: {site}, jumps{jump0,jump1}, path endings: {shiftedSite12,shiftedSite22}')

        if successes == n:
            print(f"Shift test passed for {n} trials")
        else:
            print("Shift test failed")





#    def getNeighbours(self,site):
#        neighbs = np.zeros(2*self.dim,dtype=int)
#        for dim in range(self.dim):
#            for i, jump in enumerate(self.jumps):  # Negative and positive shifts
#                neighbs[dim * 2 + i] = self.shift(site, dim, jump)
#        return neighbs
