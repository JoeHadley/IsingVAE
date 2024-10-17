from header import *

## Cv test for Wolff algorithm




numTemperatures = 50
roundsTotal = 100
updatesInARound = 10


temperatures = np.linspace(0.1,3,numTemperatures)

CArray = np.zeros(numTemperatures)
CError = np.zeros(numTemperatures)

tensor = torch.zeros([20,20])
tensor = scramble(tensor)




for i in range(numTemperatures):
  energies = np.zeros(roundsTotal)
  sqenergies = np.zeros(roundsTotal)
  for j in range(roundsTotal):
    for k in range(updatesInARound):
      tensor = wolffUpdate(tensor,temperatures[i])

    energy = getEnergy(tensor)
    energies[j] = energy
    sqenergies[j] = energy*energy


  averageEnergy = np.mean(energies)
  error = np.std(energies)/np.sqrt(roundsTotal)
  averageSqenergy = np.mean(sqenergies)
  sqerror = np.std(sqenergies)/np.sqrt(roundsTotal)





  CArray[i] = (averageSqenergy - averageEnergy*averageEnergy)/temperatures[i]

  CError[i] = np.sqrt(sqerror*sqerror + 4*averageEnergy*error*error)/temperatures[i]


  print("Done temperature ", temperatures[i])


plt.plot(temperatures,CArray)
plt.errorbar(temperatures,CArray,CError)






#plt.imshow(tensor.numpy(), cmap='gray', vmin=-1, vmax=1)

