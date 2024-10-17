# Load and preprocess training and validating data

from header import *



temperature_number = 1
sample_number = 2500
side_length = 8
training_data_number = temperature_number*sample_number

min_temp = 2.1
max_temp = 2.1

data_path = "C:/Users/jjhadley/Documents/Projects/Ising/Data/L=8/"

training_data_string = data_path +'training'
testing_data_string = data_path +'testing'


training_data, training_labels, training_temps = load_and_preprocess_data( training_data_string+'Data.dat', training_data_string+'Labels.dat', training_data_string+'TNumbers.dat', training_data_number, side_length)
training_data, training_labels, training_temps = shuffle_data(training_data, training_labels, training_temps)

testing_data, testing_labels, testing_temps = load_and_preprocess_data(testing_data_string+'Data.dat', testing_data_string+'Labels.dat', testing_data_string+'TNumbers.dat', training_data_number, side_length)
testing_data, testing_labels, testing_temps = shuffle_data(testing_data, testing_labels, testing_temps)



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
batch_size = 64
ising_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)







# Instantiate the model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(device=device).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define the loss function
def loss_function(x, x_hat, mean, logvar):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reproduction_loss + KLD


# Assuming you have train_loader defined
# train(model, optimizer, epochs=50, device=device)

def train(model, optimizer, epochs, device, data_loader):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(ising_loader):  # Adjusted to only unpack x
            x_dim = 8 * 8  # Ensure this matches your data's dimensions
            x = x.view(x.size(0), x_dim).to(device)  # Use x.size(0) for the current batch size

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"\tEpoch {epoch + 1}\tAverage Loss: {overall_loss / len(ising_loader.dataset):.4f}")
    return overall_loss

# Train the model
train(model, optimizer, epochs=100, device=device, data_loader=ising_loader)

torch.save(model.state_dict(), "C:/Users/jjhadley/Documents/Projects/VAE/VAE_model.pth")