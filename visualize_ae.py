import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from safetensors.torch import load_file

# --- Step 1: Re-define the NEW Model Architecture ---
class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- Step 2: Load the Model and Weights ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ConvolutionalAutoencoder()
state_dict = load_file("autoencoder_model.safetensors", device=str(device))
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("✅ Convolutional model loaded successfully from safetensors file.")

# --- Step 3: Load Data and Visualize ---
transform = transforms.ToTensor()
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

with torch.no_grad():
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    reconstructed = model(images)

# --- Plotting ---
def imshow(img):
    img = img.cpu().numpy()
    plt.imshow(img.squeeze(), cmap='gray')

plt.figure(figsize=(20, 4))
print("Displaying Original and Reconstructed Images:")
for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    imshow(images[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 4: ax.set_title("Original Images", fontsize=16)

    ax = plt.subplot(2, 10, i + 11)
    imshow(reconstructed[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 4: ax.set_title("Reconstructed Images", fontsize=16)
plt.show()
