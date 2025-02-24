import numpy as np
import random as r
import matplotlib.pyplot as plt
import time
import struct
import base64
import os
class Lattice:
    def __init__(self, latdims, initConfig=None,m=1,l=0, dMax0 = 1,warmCycles = None, historyLimit = int(1e6), observableFuncName = None, recordWhileWarming = False,shuffle = False):
        self.latdims = latdims
        self.Ntot = np.prod(self.latdims)
        self.dim = len(self.latdims)
        self.lat = np.random.normal(0, 1, self.Ntot) if initConfig is None else np.array(initConfig)
        self.addressList = np.arange(self.Ntot)
        self.shuffle = shuffle

        self.dMax = dMax0
        self.m = m
        self.l = l
        self.previousLat = self.lat.copy()
        self.previousMove = self.previousMove = np.zeros(2)

        self.jumps = np.zeros(2,dtype=int)
        self.jumps[0] = -1
        self.jumps[1] = 1
        


        self.observables = {
            "phi4": lambda: self.expectation(func=lambda x: x**4),
            "phi2": lambda: self.expectation(func=lambda x: x**2),
            "action": self.findAction,
            "phiBar": self.expectation,  # Default expectation

            "empty": lambda: 0,
        }




        self.historyLimit = historyLimit
        self.history = np.zeros(historyLimit)
        self.historyCount = 0
        self.historyLimitReached = False



        self.recording = observableFuncName is not None
        self.observableFuncName = observableFuncName
        self.observableFunc = self.observables.get(self.observableFuncName, self.observables["empty"])
        self.recordWhileWarming = recordWhileWarming if self.recording else False





        self.vec = np.ones(self.dim + 1)*self.Ntot

        for v in range(self.dim):
            self.vec[v+1] = self.vec[v]/latdims[v]

        self.warming = True
        if warmCycles is not None:
            self.metroCycles(warmCycles)
        self.warming = False




    
    def recordObservable(self,value = None):
        if value == None:
            value = self.observableFunc()

        if self.historyCount < self.historyLimit:
            self.history[self.historyCount] = value
            self.historyCount += 1
        else:
            print("Observable History Limit Reached")
            self.historyLimitReached = True # Just warn once

    def returnHistory(self):
        return self.history[0:self.historyCount]

    def scrambleLattice(self):
        for n in range(self.Ntot):
            self.lat[n] = r.gauss(0,1)


    def show(self):
        showLat = np.reshape(self.lat,self.latdims)
        print(showLat)
    
    def shuffleList(self):
        
        indices = np.random.permutation(self.Ntot)
        self.addressList = self.addressList[indices]
    
    def expectation(self,func=None):
        
        # If func is None, use identity
        if func is None:
            func = lambda x: x

        M = 0
        for n in range(self.Ntot):
            M += func(self.lat[n])

        M = M/self.Ntot
        return M

    def shift(self,site,dim,jump):
        shiftedSite = int(self.vec[dim]* (site //self.vec[dim]) + (site + jump * self.vec[dim+1] + self.Ntot) % (self.vec[dim]))
        return shiftedSite
    


    




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

    def getNeighbours(self,site):
        neighbs = np.array([self.shift(site, dim, jump) for dim in range(self.dim) for jump in self.jumps], dtype=int)
        return neighbs

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

    def findAction(self):
        S = 0
        dS = 0
        
        for n in range(self.Ntot):
            dS = 0
            neighbTotal = 0
            for d in range(self.dim):
                neighbTotal += self.lat[self.shift(n,d,1)]
            
            dS += (self.dim + (self.m**2)/2)*self.lat[n]**2 - self.lat[n]*neighbTotal + (self.l / 24) * self.lat[n]**4 
            
            S += dS
        return S

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


    def actionChange(self, address,d):


        neighbours = self.getNeighbours(address)

        

        neighbSum = 0
        for j in range(2*self.dim):
            n = int(neighbours[j])
            neighbSum += self.lat[n]

        dS = d*( 2*self.lat[address]*(2+ self.m*self.m/2) - neighbSum  ) \
        + d*d*(2 + self.m*self.m/2 )


        dSl = self.l*( \
        + d*np.power(self.lat[address],3)/6  \
        + d*d*np.power(self.lat[address],2)/4 \
        + d*d*d*self.lat[address]/6 \
        + d*d*d*d/24 )

        return dS +dSl




    def metroUpdate(self,n):



        d = r.gauss(0,self.dMax)

        

        dS = self.actionChange(n,d)

        boltFactor = np.exp(-dS)

        p = min(1,boltFactor)

        roll = r.uniform(0,1)


        # Apply previous move to previousLat
        self.previousLat[int(self.previousMove[0])] += self.previousMove[1]

        # Apply this move to current lattice

        update = 0
        if roll <= p:
            update = d

        self.lat[n] += update
        self.previousMove[0] = n
        self.previousMove[1] = update # Is sometimes 0  
            #self.previousMove[0] = True


        
        

        if self.recording and (not self.warming or self.recordWhileWarming) and not self.historyLimitReached:

            self.recordObservable()


        
        
    
    def metroCycle(self):
        if self.shuffle:
            self.shuffleList()


        for i in range(self.Ntot):
            n = self.addressList[i]
            self.metroUpdate(n)

    def metroCycles(self,cycles):
        for c in range(cycles):
            self.metroCycle()

    def twoPointCorr(self):
    
        C = 0


        
        CArray = np.zeros(self.Ntot**2)
        number = 0

        for n1 in range(self.Ntot):
            for n2 in range(self.Ntot):
                CArray[number] = self.lat[n1]*self.lat[n2]
                number += 1

        C = np.mean(CArray)
        CError = np.std(CArray)/(self.Ntot-1)
        return C, CError
    
    def twoPointTimeCorr(self, configNumber, interconfigCycles):

        timesteps, spacesteps = self.latdims
        
        GCMatrix = np.zeros((timesteps+1,configNumber))

        for i in range(configNumber):
            self.metroCycles(interconfigCycles)
            
            for tau2 in range(timesteps+1):
                tau = tau2 % timesteps
                corr = 0.0
                
                for t2 in range(timesteps): # Here I don't replace the +1
                    t = t2%timesteps
                    for x in range(spacesteps):
                        for y in range(spacesteps):
                            n1 = t * spacesteps + x
                            n2 = ((t + tau) % timesteps) * spacesteps + y
                            corr += self.lat[n1] * self.lat[n2]
                

                corr /= (spacesteps * spacesteps * timesteps)
                GCMatrix[tau2,i] += corr

            print("Done config ",i, "/",configNumber)

        # Average over configurations
        GCArray = np.mean(GCMatrix, axis=1)
        GCErrors = np.std(GCMatrix, axis=1)/np.sqrt(configNumber-1)

        return GCArray, GCErrors
    

    def writeConfig(self,filename = "output.bin"):
        #data = self.lat
        #data.tofile(filename)
        
        #with open(filename, mode) as file:
        #    file.write(",".join(map(str, self.lat)) + "\n")  # Write as a single line

        binary_data = self.lat.tobytes()
        encoded_data = base64.b64encode(binary_data).decode("utf-8")
        
        mode = "a" if os.path.exists(filename) else "w"
        with open(filename, mode) as file:
            file.write(encoded_data + "\n")  # Write as a single line







    def readConfig(self, filename="output.bin", copyToLat=True, line_number=0):
        configs = []

        with open(filename, "r") as file:
            for i, line in enumerate(file):
                # Decode Base64 and convert back to NumPy array
                binary_data = base64.b64decode(line.strip())
                configs.append(np.frombuffer(binary_data, dtype=np.float64))

                # Stop reading early if the desired line is reached
                if i == line_number:
                    break  # No need to read the entire file

        # Ensure the requested line exists
        if line_number >= len(configs):
            raise IndexError(f"Line number {line_number} is out of range (max {len(configs)-1}).")

        if copyToLat:
            self.lat = configs[line_number]  # Use line_number instead of copyToLat

        return configs[line_number]


    #def createTestData(self):
    #    
    #    latDimString = arr_str = "x".join(map(str, self.latdims))
    #    topString = latDimString + "\n"
