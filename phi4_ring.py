import numpy as np
import random as r
import matplotlib.pyplot as plt

def shuffle_list(list):
    num_rows = len(list)
    indices = np.random.permutation(num_rows)
    list = list[indices]

    return list









def findAction(lat, latdims, m=1, l=0):
    S = 0

    for n in range(Ntot):
        dS = 0.5 * ( lat[shift(latdims,n,0,1)] - lat[n]) ** 2  + 0.5 * ( lat[shift(latdims,n,1,1)] - lat[n]) ** 2 + 0.5 * m**2 * lat[n]**2   + (l / 24) * lat[n]**4 
        
        S += dS
    return S

def actionChange(lat, latdims, address,d,  m=1, l=0):
    #print("actionChange called. lat: ", lat, ", latdims: ", latdims, ", address: ", address, ",d: ",d)
    dimension = len(latdims)

    neighbours = getNeighbours(latdims,address)

    

    neighbSum = 0
    for j in range(2*dimension):
        n = int(neighbours[j])
        neighbSum += lat[n]

    dS = d*( 2*lat[address]*(2+ m*m/2) - neighbSum  ) \
    + d*d*(2 + m*m/2 )


    dSl =  d*np.power(lat[address],3)*l/6  \
    + d*d*np.power(lat[address],2)*l/4 \
    + d*d*d*(lat[address]*l/6) \
    + d*d*d*d*(l/24)

    return dS

def expectation(lat,func=None):
    # Assume the lattice is a 1D ring
    M = 0
    N = len(lat)
    
    # If func is None, use identity
    if func is None:
        func = lambda x: x


    for i in range(N):
        M += func(lat[i])

    M = M/N
    return M





def twoPointCorr(lat1, lat2=None):
   
    C = 0

    Ntot = len(lat1)

    # If ring2 is None, use ring1 for both
    if lat2 is None:
        lat2 = np.copy(lat1)
    
    CArray = np.zeros(Ntot*Ntot)
    number = 0

    for n1 in range(Ntot):
        for n2 in range(Ntot):
            CArray[number] = lat1[n1]*lat2[n2]
            number += 1

    C = np.mean(CArray)
    CError = np.std(CArray)/Ntot
    return C, CError





def getCoords(n, sideLength):
    row = int(n//sideLength)
    col = int(n%sideLength)
    return row,col


def metropolisUpdate(lat,latdims,n,dMax):


    sideLength = lat.shape[0]
    
    d=r.uniform(-dMax,dMax)
    

    #print("lat length: ", len(lat))
    dS = actionChange(lat,latdims,n,d)



    p = min(1,np.exp(-dS))

    roll = r.uniform(0,1)

    if roll <= p:
        lat[n] = lat[n] + d
    return lat


def warmLattice(lat,latdims,g0,readout = False, returnArray = False, flip = False):
    dMax = 1
    dMaxes = np.zeros(g0)
    Ntot = len(lat)

    for i in range(g0):
        list = shuffle_list(addressList)
        for j in range(Ntot):

            n = list[j]
                   
            lat2 = np.copy(lat)
    
            d=r.uniform(-dMax,dMax)

    
            lat2[n] = lat[n] + d

            dS = findAction(lat2,latdims) - findAction(lat,latdims)



            p = min(1,np.exp(-dS))
            roll = r.uniform(0,1)

            # Adjust dMax to try and achieve a probability of about 80%
            if p < 0.78:
                dMax = dMax*0.95
            elif p > 0.82:
                dMax = dMax*1.05

            flip = False
            if flip:

                if roll <= p:
                    lat[n] = lat[n] + d
                
        if readout:

            print("Run  ", i, "/",g0,": dMax = ",dMax, "d = ", d, "p = ",p, "dS = ", dS)

        dMaxes[i]=dMax

    if returnArray:
        return dMaxes
    else:
        return dMax






def getNeighbours(latdims,site):
    D = len(latdims)
    Ntot = np.product(latdims)
    vec = np.ones(D+1)*Ntot

    for v in range(D):
        vec[v+1] = vec[v]/latdims[v]

    neighbs = np.zeros(2*D)
    for i in range(2*D):
        d = int(np.ceil((i+1)/2))-1

        neighbSite = vec[d]* (site //vec[d]) + (site + ((-1)**i) * vec[d+1] + Ntot) % (vec[d])
        neighbs[i] = int(neighbSite)
        #print("d is ", d, ", neighbour is ",neighbSite)
    return neighbs

def shift(latdims,site,dim,s):
    D = len(latdims)
    Ntot = np.product(latdims)
    vec = np.ones(D+1)*Ntot

    for v in range(D):
        vec[v+1] = vec[v]/latdims[v]




    d = dim

    neighbSite = int(vec[d]* (site //vec[d]) + (site + s * vec[d+1] + Ntot) % (vec[d]))

    return neighbSite

##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################


latdims = np.array((10,10)) # convention is t,x,y,z

Ntot = np.product(latdims)
lat = np.zeros(Ntot)




# Number of warming sweeps
g0 = 10

sweeps = 2000

# Create address list to permute each sweep
addressList = np.zeros(Ntot,dtype=int)
for i in range(Ntot):
    addressList[i] = i

# Initialise lattice

lat0 = np.random.uniform(low=-1, high=1, size=Ntot)



# Initialise lattice through computer time


latm = np.zeros((sweeps,Ntot))
latm[0,:] = lat0





# Find a good value for dMax
dMax = warmLattice(lat,latdims,g0)

print("dmax is", dMax)


corrArray = np.zeros(int(sweeps/10))
corrErrorArray = np.zeros(int(sweeps/10))
number = 0




for j in range(Ntot):
    # Dealing with each site in a random order
    n = addressList[j]
    lat0 = metropolisUpdate(lat0,latdims,n,dMax)


#print(lat0)

for i in range(sweeps):
    list = shuffle_list(addressList)
    lat = latm[i,:] # Work with that slice 

    for j in range(Ntot):
        # Dealing with each site in a random order

        n = addressList[j]

        lat = metropolisUpdate(lat,latdims,n,dMax)
    
    if (i%10) == 0:

        correlator, corrError = twoPointCorr(lat)
        corrArray[number] = correlator
        corrErrorArray[number] = corrError
        number+=1


    if i != sweeps-1:

        latm[i+1,:] = lat # Save that configuration to the matrix
        
    print("Done sweep ", i+1,"/",sweeps)





print(corrArray)

averageCorrArray = np.mean(corrArray)
print("Simple correlator mean: ", averageCorrArray)

standardDeviation = np.std(corrArray)
#print(standardDeviation)
errorOnMean = standardDeviation/np.sqrt(len(corrArray))
print("Error on mean: ", errorOnMean)




# Weighted Mean 

corrNumber = len(corrErrorArray)

denominator =0
numerator = 0
for i in range(corrNumber):
    ddenom = 1/((corrErrorArray[i])**2)
    denominator += ddenom

    dnumer = corrArray[i]*ddenom
    numerator += dnumer

weightedCorrMean = numerator/denominator
print("Weighted Correlator Mean: ",weightedCorrMean)

# Uncertainty on weighted mean

wMeanError = np.sqrt(1/denominator)
print("Error on weighted mean: ", wMeanError)
