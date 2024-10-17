from header import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(device=device).to(device)

# Load the VAE weights
model.load_state_dict(torch.load("C:/Users/jjhadley/Documents/Projects/VAE/VAE_model.pth",weights_only=True))

largeSize = 20
smallSize = 8
tensor = torch.zeros([largeSize, largeSize], dtype=torch.float32)

tensor = scramble(tensor)

tx = r.randint(0,largeSize)
ty = r.randint(0,largeSize)

smallTensor = tensorCut(tensor,smallSize, [tx,ty])

input_image = smallTensor.view(-1)



generatedImage, mean = model.generate_from_image(input_image)

generatedImage = generatedImage.view(8, 8)

#generatedImage = tensorRound(generatedImage)

test = tensorRound(smallTensor)


# Plotting
#plt.figure(figsize=(8, 4))  # Set the figure size

# Create a subplot with 1 row and 2 columns
f, axarr = plt.subplots(1, 3)


# Plot the original tensor
axarr[0].imshow(tensor.detach().numpy(), cmap='gray', vmin=0, vmax=1)  # Use appropriate colormap and normalization
axarr[0].set_title('Original 20x20 Configuration')
axarr[0].axis('off')  # Hide axes for better visualization

# Plot the 8x8 patch
axarr[1].imshow(smallTensor.detach().numpy(), cmap='gray', vmin=0, vmax=1)  # Use appropriate colormap and normalization
axarr[1].set_title('8x8 Patch')
axarr[1].axis('off')  # Hide axes for better visualization

# Plot the generated image
axarr[2].imshow(generatedImage.detach().numpy(), cmap='gray', vmin=0, vmax=1)  # Use appropriate colormap and normalization
axarr[2].set_title('Generated Image')
axarr[2].axis('off')  # Hide axes for better visualization

# Display the plots
plt.tight_layout()
plt.show()


