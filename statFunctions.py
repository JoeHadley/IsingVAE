import numpy as np

def specific_heat(data, beta, N_dof):
    mean = np.mean(data)
    mean_sq = np.mean(data**2)
    return beta**2 * (mean_sq - mean**2) / N_dof


def jackknife_bins(data, bin_size, beta, N_dof):
    n = len(data)
    n_bins = n // bin_size
    # reshape into bins
    binned = data[:n_bins*bin_size].reshape(n_bins, bin_size)
    jk_vals = []
    for i in range(n_bins):
        reduced = np.delete(binned, i, axis=0).ravel()
        jk_vals.append(specific_heat(reduced, beta, N_dof))
    jk_vals = np.array(jk_vals)
    mean = np.mean(jk_vals)
    var = (n_bins-1)/n_bins * np.sum((jk_vals - mean)**2)
    return mean, np.sqrt(var)


def autocorr_func(x, max_lag):
    x = np.asarray(x)
    x = x - np.mean(x)
    n = len(x)
    result = np.correlate(x, x, mode='full')
    acf = result[n-1:n+max_lag] / result[n-1]
    return acf

def integrated_autocorr_time(x, max_lag=None):
    if max_lag is None:
        max_lag = len(x) // 10  # safe cutoff
    acf = autocorr_func(x, max_lag)
    tau_int = 0.5 + np.sum(acf[1:])
    return tau_int, acf



# Example: choose bin size >> autocorrelation time

#bin_size = 2 #TODO Get autocorrelation time from data
#C, C_err = jackknife_bins(actions, bin_size, beta, Ntot)
#print("Specific heat:", C, "+/-", C_err)



#actions = simulation.observer.returnHistory()
#tau_int, acf = integrated_autocorr_time(actions, max_lag=200)
#print("Estimated tau_int:", tau_int)

#import matplotlib.pyplot as plt

#lags = np.arange(len(acf))
#plt.plot(lags, acf, marker='o')
#plt.xlabel("Lag")
#plt.ylabel("Autocorrelation")
#plt.title("Autocorrelation function of action")
#plt.show()