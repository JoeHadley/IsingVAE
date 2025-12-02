








GCArray, GCErrors = lat.twoPointTimeCorr(correlatorConfigs,interconfigCycles)
constant = GCArray[0]


folder = "Results\\10x10\\"

#filePath = folder + "GCArray"+"10^"+str(exp1)+"10^"+str(exp2)+ "10^"+str(exp3)+".txt"
np.savetxt(filePath, GCArray)

filePath = folder + "GCErrors"+"10^"+str(exp1)+"10^"+str(exp2)+ "10^"+str(exp3)+".txt"
np.savetxt(filePath, GCErrors)




# Analytic calculation

analyticResults = getAnalyticCorrelator(lattice,m)


print(analyticResults)





xAxis = np.arange(latdims[0]+1)
C, CError = lat.twoPointCorr()

plt.errorbar(xAxis,GCArray,GCErrors,label="Monte Carlo")
plt.title(f"{T}x{L} lattice, {pregameWarmCycles} warming cycles, {correlatorConfigs} configs, {interconfigCycles} cycles between configs")
plt.xlabel("Tau")





plt.plot(analyticResults,linestyle="dashed",label="Analytical")


plt.legend()

plt.show()