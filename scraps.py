
training_data_string = data_path +'training'
testing_data_string = data_path +'testing'


training_data, training_labels, training_temps = load_and_preprocess_data( training_data_string+'Data.dat', training_data_string+'Labels.dat', training_data_string+'TNumbers.dat', training_data_number, side_length)
training_data, training_labels, training_temps = shuffle_data(training_data, training_labels, training_temps)

testing_data, testing_labels, testing_temps = load_and_preprocess_data(testing_data_string+'Data.dat', testing_data_string+'Labels.dat', testing_data_string+'TNumbers.dat', training_data_number, side_length)
testing_data, testing_labels, testing_temps = shuffle_data(testing_data, testing_labels, testing_temps)






def readConfig(self, filename = "output.bin",copyToLat = True):

    configs = []
    with open(filename, "r") as file:
        for line in file:
            # Decode Base64 and convert back to NumPy array
            binary_data = base64.b64decode(line.strip())
            configs.append(np.frombuffer(binary_data, dtype=np.float64))




    if copyToLat:
        self.lat = configs[copyToLat]

    return configs

#def forward(self, x):
#    mean, logvar = self.encode(x)
#    z = self.reparameterization(mean, logvar)
#    x_hat = self.decode(z)
#    return x_hat, mean, logvar



log_alpha = (
            log_p_phi_prime - log_p_phi
            + log_q_zF - log_q_zF
            + log_det_jF - log_det_jB
        )


def getDetTerm(input_z):
    jacobian = myVAE.compute_decoder_jacobian(input_z)  # Compute the Jacobian
    JTJ = jacobian.T @ jacobian  # Compute Jacobian transpose @ Jacobian
    detTerm = torch.sqrt(torch.linalg.det(JTJ))  # Compute the square root of the determinant
    return detTerm

jacobian = myVAE.compute_decoder_jacobian(torch.tensor([1.0, 2.0]))  # Example input for Jacobian computation
#jac_norm = jacobian.norm()  # Compute the norm of the Jacobian
JTJ = jacobian.T @ jacobian
detTerm = torch.sqrt(torch.linalg.det(JTJ))   
#print("Jacobian:", jacobian)
#print("Jacobian Norm:", jac_norm)
print("detTerm:", detTerm)  # Print the product of Jacobian transpose and Jacobian
print("detTerm from getDetTerm:", getDetTerm(torch.tensor([1.0, 2.0])))  # Example input for determinant term computation
print(myVAE.compute_decoder_jacobian(torch.tensor([1.0, 2.0])))  # Example input for Jacobian computation

print(myVAE.jacobian_norm(torch.tensor([1.0, 2.0]), myVAE.decode))  # Example input for Jacobian norm computation






    def compute_acceptance_probability(self, input_phi, output_phi):
        """
        Compute the acceptance probability for the proposed update.
        :param input_phi: Current field value
        :param output_phi: Proposed new field value
        :return: Acceptance probability
        """
        

        log_q_zF = self.log_prob_q(output_phi, input_phi)
        log_q_zB = self.log_prob_q(input_phi, output_phi)
        log_det_jF = torch.log(self.compute_jacobian_term(output_phi))
        log_det_jB = torch.log(self.compute_jacobian_term(input_phi))
        log_alpha = ( log_q_zF - log_q_zB - log_det_jF + log_det_jB
        )

        acceptance_prob = torch.exp(log_alpha).clamp(max=1.0)
        return acceptance_prob.item()  # Return as a scalar value






# Watch out, these may be needed again


    def log_prior(self,phi, mu0, sigma0):
            """Log Gaussian prior: N(mu0, sigma0^2 I)"""
            d = phi.shape[0]
            var0 = sigma0**2
            diff = phi - mu0
            log_term = torch.log(torch.tensor(2.0 * torch.pi * var0, device=diff.device))
            logp = -0.5 * (d * log_term + (diff ** 2).sum() / var0)
            return logp

    def log_likelihood(self,phi, data, sigma_y):
        """
        Gaussian likelihood: p(data | phi) = N(data | g(phi), sigma_y^2 I)
        - forward_model(phi): predicts mean of data given phi
        """
        
        pred = self.forward(phi)  # Forward pass to get predictions

        
        var_y = sigma_y**2
        diff = data - pred
        d = data.shape[0]
        log_term = torch.log(torch.tensor(2.0 * torch.pi * var_y, device=diff.device))
        logl = -0.5 * (d * log_term + (diff ** 2).sum() / var_y)
        return logl

    def log_p(self,phi, data, mu0, sigma0, sigma_y):
        """Log posterior up to constant"""
        log_p = self.log_likelihood(phi, data, sigma_y) + self.log_prior(phi, mu0, sigma0)
        return log_p