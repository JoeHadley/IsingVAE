# Load and preprocess training and validating data

from header import *




sample_number = 100
side_length = 10
training_data_number = sample_number

training_data_path = "C:/Users/Joe/Documents/Projects/IsingVAE/TrainingData.bin"
testing_data_path = "C:/Users/Joe/Documents/Projects/IsingVAE/TestingData.bin"


training_data = processDataPhi4(training_data_path,side_length)
testing_data = processDataPhi4(training_data_path,side_length)




# My data in a numpy array
numpy_data = training_data  # Example data with 2500 samples of 8x8 images

# Convert to PyTorch tensor
tensor_data = torch.tensor(numpy_data, dtype=torch.float32)

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, tensor_data):
        self.data = tensor_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

# Create a dataset object
dataset = CustomDataset(tensor_data)

# Create a DataLoader object
batch_size = 100
phi4_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)







# Instantiate the model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(device=device).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define the loss function
def loss_function(x, x_hat, mean, logvar):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reproduction_loss + KLD


# Assuming you have train_loader defined
# train(model, optimizer, epochs=50, device=device)

def train(model, optimizer, epochs, device, data_loader):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(data_loader):  # Adjusted to only unpack x
            x_dim = side_length * side_length  # Ensure this matches your data's dimensions
            #x = x.view(x.size(0), x_dim).to(device)  # Use x.size(0) for the current batch size

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"\tEpoch {epoch + 1}\tAverage Loss: {overall_loss / len(data_loader.dataset):.4f}")
    return overall_loss

# Train the model
train(model, optimizer, epochs=100, device=device, data_loader=phi4_loader)

#torch.save(model.state_dict(), "C:/Users/jjhadley/Documents/Projects/VAE/VAE_model.pth")