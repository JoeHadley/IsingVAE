import numpy as np
import random as r

class Lattice:
    def __init__(self, latdims, initConfig=None,m=1,l=0,gauss = True, dMax0 = 1):
        self.latdims = np.array(latdims)
        self.Ntot = np.prod(self.latdims)
        self.dim = len(self.latdims)
        self.lat = np.zeros(self.Ntot) if initConfig is None else np.array(initConfig)
        self.addressList = np.arange(self.Ntot)
        self.dMax = dMax0
        self.gauss = gauss
        self.m = m
        self.l = l


        self.vec = np.ones(self.dim + 1)*self.Ntot

        for v in range(self.dim):
            self.vec[v+1] = self.vec[v]/latdims[v]
    
    def scrambleLattice(self):
        for n in range(self.Ntot):
            if self.gauss:
                self.lat[n] = r.gauss(0,1)
            else:
                self.lat[n] = r.uniform(-1,1)

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


        for n in range(self.Ntot):
            M += func(self.lat[n])

        M = M/self.Ntot
        return M

    def shift(self,site,dim,jump):
        shiftedSite = int(self.vec[dim]* (site //self.vec[dim]) + (site + jump * self.vec[dim+1] + self.Ntot) % (self.vec[dim]))
        return shiftedSite

    def getNeighbours(self,site):
        neighbs = np.zeros(2*self.dim)
        for i in range(2*self.dim):
            d = int(np.ceil((i+1)/2))-1

            jump = (-1)**i
            neighbs[i] = self.shift(site,d,jump)
        return neighbs



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

        return dS+dSl

    def warmLattice(self,g0,readout = False, flip = False):
        if readout:

            actionArray = np.zeros(g0)

        for i in range(g0):

            self.shuffleList()
            for j in range(self.Ntot):

                n = self.addressList[j]
                
                if self.gauss:
                    d = r.gauss(0,self.dMax)
                else:
                    d=r.uniform(-self.dMax,self.dMax)

        
                dS = self.actionChange(n,d)



                p = min(1,np.exp(-dS))
                roll = r.uniform(0,1)

                # Adjust dMax to try and achieve a probability of about 80%
                if p < 0.78:
                    self.dMax *= 0.99
                elif p > 0.82:
                    self.dMax *= 1.01

                if flip:

                    if roll <= p:
                        self.lat[n] = self.lat[n] + d
                    
            if readout:
                S = self.findAction()
                print("Run  ", i, "/",g0,": dMax = ",self.dMax, "d = ", d, "p = ",p, "dS = ", dS, "S = ",S)
                actionArray[i] = S
        if readout:
            return actionArray


    def metropolisUpdate(self,n):

        if self.gauss:
            d = r.gauss(0,self.dMax)
        else:
            d=r.uniform(-self.dMax,self.dMax)
        

        dS = self.actionChange(n,d)



        p = min(1,np.exp(-dS))

        roll = r.uniform(0,1)

        if roll <= p:
            self.lat[n] += d
    
    def metropolisCycles(self,cycles,readout = False):

        if readout:
            actionArray = np.zeros(cycles)

        

        for c in range(cycles):
            self.shuffleList()


            for i in range(self.Ntot):
                n = self.addressList[i]
                self.metropolisUpdate(n)
        
        
            if readout:
                S = self.findAction()
                print("Run  ", c, "/",cycles)
                actionArray[c] = S

        if readout:
            return actionArray

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
            self.metropolisCycles(interconfigCycles)
            
            for tau2 in range(timesteps+1):
                tau = tau2 % timesteps
                corr = 0.0
                
                for t2 in range(timesteps+1):
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


            

T = 10
L = 10

latdims2 = np.array((T,L))
lat2 = Lattice(latdims2)


lat2.scrambleLattice()
lat2.metropolisCycles(10)
print("Lattice warmed")


GCArray2, GCErrors2 = lat2.twoPointTimeCorr(1000)
xAxis = np.arange(latdims2[0]+1)
C, CError = lat2.twoPointCorr()
print(xAxis)
print(GCArray2)

plt.errorbar(xAxis,GCArray2,GCErrors2,label="Monte Carlo")
plt.title("Correlation function for 10x10 euclidean lattice between phi(x,0) and phi(y,tau)")
plt.xlabel("Tau")





# Analytic calculation



m = 1



def denominator(nx,nt):
    return 4*np.sin(np.pi*nx/L)**2 + 4*np.sin(np.pi*nt/T)**2 + m**2

def real_numerator(x1,x2,nx,t1,t2,nt):
    return np.cos(2*np.pi*nx*(x2-x1)/L + 2*np.pi*nt*(t2-t1)/T)


def imag_numerator(x1,x2,nx,t1,t2,nt):
    return np.sin(2*np.pi*nx*(x2-x1)/L + 2*np.pi*nt*(t2-t1)/T)

def anCorr(x1,x2,t1,t2):
    result = 0
    for nx in range(L):
        for nt in range(T):
            result += real_numerator(x1,x2,nx,t1,t2,nt)/denominator(nx,nt)
    return result/(L*T)



results = np.zeros(T+1)

for tau in range(T+1):

    for x in range(L):
        for y in range(L):
            for t in range(T):
                results[tau] += anCorr(x,y,t,t+tau)

results /= (L*L*T)

print(results)

plt.plot(results,linestyle="dashed",label="Analytical")

plt.legend()

plt.show()








def getCoords(n, sideLength):
    row = int(n//sideLength)
    col = int(n%sideLength)
    return row,col





def denominator(nx,nt):
    return 4*np.sin(np.pi*nx/L)**2 + 4*np.sin(np.pi*nt/T)**2 + m**2

def real_numerator(x1,x2,nx,t1,t2,nt):
    return np.cos(2*np.pi*nx*(x2-x1)/L + 2*np.pi*nt*(t2-t1)/T)


def imag_numerator(x1,x2,nx,t1,t2,nt):
    return np.sin(2*np.pi*nx*(x2-x1)/L + 2*np.pi*nt*(t2-t1)/T)

def anCorr(x1,x2,t1,t2):
    result = 0
    for nx in range(L):
        for nt in range(T):
            result += real_numerator(x1,x2,nx,t1,t2,nt)/denominator(nx,nt)
    return result/(L*T)








##############################################################
##############################################################




pregameWarmCycles = 10000
correlatorConfigs = 200
interconfigCycles = 200
gauss = True

T = 5
L = 5

latdims2 = np.array((T,L))
lat2 = Lattice(latdims2,gauss = True,dMax0=0.1)



lat2.scrambleLattice()
lat2.metropolisCycles(pregameWarmCycles)
print("Lattice warmed")


GCArray2, GCErrors2 = lat2.twoPointTimeCorr(correlatorConfigs,interconfigCycles)
constant = GCArray2[0]









# Analytic calculation

m = 1

results = np.zeros(T+1)

for tau in range(T+1):

    for x in range(L):
        for y in range(L):
            for t in range(T):
                results[tau] += anCorr(x,y,t,(t+tau)%T)

results /= (L*L*T)


#results /= results[0]
print(results)





xAxis = np.arange(latdims2[0]+1)
C, CError = lat2.twoPointCorr()
#print(xAxis)
#print(GCArray2/GCArray2[0])

plt.errorbar(xAxis,GCArray2,GCErrors2,label="Monte Carlo")
plt.title(f"Correlation function for {T}x{L} euclidean lattice between phi(x,0) and phi(y,tau)")
plt.xlabel("Tau")





plt.plot(results,linestyle="dashed",label="Analytical")


plt.legend()

plt.show()

print("Plotted")

