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
