import numpy as np
import random as r

def shuffle(list):
    num_rows = len(list)
    indices = np.random.permutation(num_rows)
    list = list[indices]

    return list


def findAction(phi,m=1,l=1):
    N = len(phi)
    S = 0
    for i in range(N):
        dS = np.power(phi[(i+1)%N] - phi[i],2)/2 +  np.power(phi[i],2)* np.power(m,2) /2 + np.power(phi[i],4)*l/24
        S = S + dS
    return S



def actionChange(phi, n, d,  m=1, l=1):
    dS = d*( 2*phi[n] - phi[(n+1)%N] - phi[(n-1)%N] + m*m*phi[n] + np.power(phi[n],3)*l/6  ) \
    + d*d*(1 + m*m /2 + np.power(phi[n],2)*l/4) \
    + d*d*d*(phi[n]*l/6) \
    + d*d*d*d*(l/24)
    return dS


# Number of sites, and values of m and lambda
N = 10
m = 1
l = 1




# Initialise ring
phi = np.random.uniform(0,1,N)



# Create address list to permute each sweep
addressList = np.zeros(N,dtype=int)
for i in range(N):
    addressList[i] = i





# Number of warming sweeps, and initial variation limit
g0=100
dMax = 1

dphi = np.full(N, 0.0, dtype=float)






def heatBathUpdate(phi,dphi,n,dMax):

    
    d=r.uniform(-dMax,dMax)

    dphi[n] = d

    phi2 = phi + dphi
    dphi[n] = 0.0

    dS = findAction(phi2) - findAction(phi)



    p = min(1,np.exp(-dS))
    roll = r.uniform(0,1)

    if roll <= p:
        phi[n] = phi[n] + d
    return phi



warm = False

if warm == True:
    for i in range(g0):
        list = shuffle(addressList)
        for j in range(N):

            n = list[j]
            
            d=r.uniform(-dMax,dMax)

            dphi[n] = d

            phi2 = phi + dphi
            dphi[n] = 0.0

            dS = findAction(phi2) - findAction(phi)



            p = min(1,np.exp(-dS))
            roll = r.uniform(0,1)

            # Adjust dMax to try and achieve a probability of about 80%
            if p < 0.78:
                dMax = dMax*0.95
            elif p > 0.82:
                dMax = dMax*1.05
            
            #if roll <= p:
            #    phi[n] = phi[n] + d


dMax = 0.8140982829337562
sweeps = 100
for i in range(sweeps):
    list = shuffle(addressList)
    for j in range(N):

        n = list[j]
        phi = heatBathUpdate(phi,dphi,n,dMax)


print(dMax)
