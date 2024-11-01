import numpy as np
import random as r
import matplotlib.pyplot as plt

def shuffle_list(list):
    num_rows = len(list)
    indices = np.random.permutation(num_rows)
    list = list[indices]

    return list


def findAction(ring,m=1,l=0):
    N = len(ring)
    S = 0
    for i in range(N):
        dS = np.power(ring[(i+1)%N] - ring[i],2)/2 +  np.power(ring[i],2)* np.power(m,2) /2 + np.power(ring[i],4)*l/24
        S = S + dS
    return S



def actionChange(phi, n, d,  m=1, l=0):
    N = len(phi)
    dS = d*( 2*phi[n] - phi[(n+1)%N] - phi[(n-1)%N] + m*m*phi[n] + np.power(phi[n],3)*l/6  ) \
    + d*d*(1 + m*m /2 + np.power(phi[n],2)*l/4) \
    + d*d*d*(phi[n]*l/6) \
    + d*d*d*d*(l/24)
    return dS

def expectation(ring,func=None):
    # Assume the lattice is a 1D ring
    M = 0
    N = len(ring)
    
    # If func is None, use identity
    if func is None:
        func = lambda x: x


    for i in range(N):
        M += func(ring[i])

    M = M/N
    return M

def twoPointCorr(ring1, ring2=None):
    # Assume the lattice is a 1D ring
    C = 0
    N = len(ring1)  # Assume equal dimensions for both rings if ring2 is provided
    
    # If ring2 is None, use ring1 for both
    if ring2 is None:
        ring2 = ring1
    
    for i in range(N):
        for j in range(N):
            dC = ring1[i]*ring2[j]
            C = C + dC

    C = C/(N*N)
    return C

def twoTimeCorr(ring1, ring2=None):
    # ring1 is the earlier time, ring2 is later

    # Assume the lattice is a 1D ring
    C = 0
    N = len(ring1)  # Assume equal dimensions for both rings if ring2 is provided
    
    # If ring2 is None, use ring1 for both
    if ring2 is None:
        ring2 = ring1
    
    for i in range(N):
        dC = ring1[i]*ring2[i]
        C = C + dC

    C = C/N
    return C

def metropolisUpdate(phi,n,dMax):

    #phi2 = np.copy(phi)
    
    d=r.uniform(-dMax,dMax)
    
    #phi2[n] = phi2[n] + d
    

    dS = actionChange(phi,n,d,1,0)



    p = min(1,np.exp(-dS))

    roll = r.uniform(0,1)

    if roll <= p:
        phi[n] = phi[n] + d
    return phi

def warmLattice(phi,g0,readout = False):
    dMax = 1
    dMaxes = np.zeros(g0)
    for i in range(g0):
        list = shuffle_list(addressList)
        for j in range(N):

            n = list[j]
            
            phi2 = np.copy(phi)
    
            d=r.uniform(-dMax,dMax)

    
            phi2[n] = phi2[n] + d

            dS = findAction(phi2,l=0) - findAction(phi,l=0)



            p = min(1,np.exp(-dS))
            roll = r.uniform(0,1)

            # Adjust dMax to try and achieve a probability of about 80%
            if p < 0.78:
                dMax = dMax*0.95
            elif p > 0.82:
                dMax = dMax*1.05

            flip = True
            if flip:

                if roll <= p:
                    phi[n] = phi[n] + d
                
        if readout:

            print("Run  ", i, "/",g0,": dMax = ",dMax, "d = ", d, "p = ",p, "dS = ", dS)
        #dMaxes[i]=dMax
    return dMax

def timeCorrelator(phi1, phi2):
    var1 = twoTimeCorr(phi1,phi2)
    var2 = expectation(phi2)*expectation(phi1)
    # var2 = pow(expectation(phi2),2) # If I don't expect the mean to drift
    var3 = expectation(phi1,func = lambda x: x**2)

    timeCorr = (var1 - var2)/(var3-var2)


    return timeCorr





# Number of sites, and values of m and lambda
N = 1000
m = 1
l = 1

# Number of warming sweeps
g0 = 10

sweeps = 1000

# Create address list to permute each sweep
addressList = np.zeros(N,dtype=int)
for i in range(N):
    addressList[i] = i

# Initialise ring
phi0 = np.random.uniform(-1,1,N)

# Initialise ring through time
phim = np.zeros((sweeps,N))
phim[0,:] = phi0


# Find a good value for dMax
#dMax = warmLattice(phi0,g0)
dMax = 0.8140982829337562 # A good value found using warmLattice()


corrArray = np.zeros(50)
number = 0

for i in range(sweeps):
    list = shuffle_list(addressList)
    phi = phim[i,:] # Work with that row 
    
    for j in range(N):
        # Dealing with each site in a random order

        n = list[j]
        phi = metropolisUpdate(phi,n,dMax)
    
    if i != sweeps-1:

        phim[i+1,:] = phi # Save that configuration to the matrix
    print("Done sweep ", i+1,"/",sweeps)



    if i%20 == 0:
        corrArray[number] = twoPointCorr(phi)
        number += 1


print(np.mean(corrArray))




# Compute time correlator for each sweep
#T = np.zeros(sweeps)
#for i in range(500):
#    T[i] = timeCorrelator(phim[0,:], phim[i,:])
#    print("Time correlator ", i, " done")
#plt.plot(T)
#plt.hlines(0,0,sweeps, colors="k",linestyles="dashed")
#plt.title("Autocorrelation Over Time")
#plt.ylabel("Normalised Correlation")
#plt.xlabel("Sweep Number")
#plt.show()




