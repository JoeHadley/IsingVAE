    def writeConfig(self,filename = "output.bin"):
        #data = self.lat
        #data.tofile(filename)
        
        #with open(filename, mode) as file:
        #    file.write(",".join(map(str, self.lat)) + "\n")  # Write as a single line

        binary_data = self.lat.tobytes()
        encoded_data = base64.b64encode(binary_data).decode("utf-8")
        
        mode = "a" if os.path.exists(filename) else "w"
        with open(filename, mode) as file:
            file.write(encoded_data + "\n")  # Write as a single line







    def readConfig(self, filename="output.bin", copyToLat=True, line_number=0):
        configs = []

        with open(filename, "r") as file:
            for i, line in enumerate(file):
                # Decode Base64 and convert back to NumPy array
                binary_data = base64.b64decode(line.strip())
                configs.append(np.frombuffer(binary_data, dtype=np.float64))

                # Stop reading early if the desired line is reached
                if i == line_number:
                    break  # No need to read the entire file

        # Ensure the requested line exists
        if line_number >= len(configs):
            raise IndexError(f"Line number {line_number} is out of range (max {len(configs)-1}).")

        if copyToLat:
            self.lat = configs[line_number]  # Use line_number instead of copyToLat

        return configs[line_number]


    #def createTestData(self):
    #    
    #    latDimString = arr_str = "x".join(map(str, self.latdims))
    #    topString = latDimString + "\n"
