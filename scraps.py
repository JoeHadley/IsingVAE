
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