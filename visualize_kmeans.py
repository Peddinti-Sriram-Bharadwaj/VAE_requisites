import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from safetensors.torch import load_file
import joblib

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid()

                )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


device = torch.device("mps")
model = ConvolutionalAutoencoder()
state_dict = load_file("autoencoder_model.safetensors", device=str(device))
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Autoencoder model loaded.")


kmeans = joblib.load("kmeans_codebook.joblib")

codebook = torch.from_numpy(kmeans.cluster_centers_).float().to(device)
print("k-means codebook loaded.")

transform = transforms.ToTensor()
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10 ,shuffle=True)

with torch.no_grad():
    images, _ = next(iter(test_loader))
    images = images.to(device)

    original_reconstructions = model(images)

    continuous_latents = model.encoder(images)
    original_shape = continuous_latents.shape

    continuous_latents_flat = continuous_latents.view(original_shape[0], -1)
    quantized_indices = kmeans.predict(continuous_latents_flat.cpu().numpy())
    quantized_latents_flat = codebook[quantized_indices]
    quantized_latents = quantized_latents_flat.view(original_shape)
    quantized_reconstructions = model.decoder(quantized_latents)

def imshow(img, ax):
    img = img.cpu().numpy()
    ax.imshow(img.squeeze(), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

fig, axes = plt.subplots(3, 10, figsize=(20, 6))
fig.suptitle('Original vs. Reconstructed vs. Quantized', fontsize=20)

for i in range(10):
    # Display original image
    imshow(images[i], axes[0, i])
    if i == 0: axes[0, i].set_ylabel("Original", fontsize=16)

    # Display normal reconstruction
    imshow(original_reconstructions[i], axes[1, i])
    if i == 0: axes[1, i].set_ylabel("Reconstructed", fontsize=16)

    # Display quantized reconstruction
    imshow(quantized_reconstructions[i], axes[2, i])
    if i == 0: axes[2, i].set_ylabel("Quantized", fontsize=16)

plt.show()


