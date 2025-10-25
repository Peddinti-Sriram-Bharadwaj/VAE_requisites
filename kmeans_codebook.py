import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from safetensors.torch import load_file
import joblib

# --- Step 1: Load the Trained Autoencoder ---
# CORRECT: Using the ConvolutionalAutoencoder definition
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

# Load the weights into the correct model structure
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ConvolutionalAutoencoder()
state_dict = load_file("autoencoder_model.safetensors", device=str(device))
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("✅ Trained convolutional autoencoder model loaded.")


# --- Step 2: Generate Latent Vectors for the whole dataset ---
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
full_data_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)

latent_vectors = []
with torch.no_grad():
    for images, _ in full_data_loader:
        images = images.to(device)
        encoded = model.encoder(images)
        latent_vectors.append(encoded.cpu())

latent_vectors = torch.cat(latent_vectors, dim=0)
print(f"Original latent vectors shape: {latent_vectors.shape}")

# This will now correctly be (60000, 32, 7, 7)
# And the flattened shape will be (60000, 1568)
num_images = latent_vectors.shape[0]
latent_vectors_flat = latent_vectors.view(num_images, -1).numpy()
print(f"Flattened latent vectors shape: {latent_vectors_flat.shape}")


# --- Step 3: Train k-Means to create the "Codebook" ---
num_clusters = 256
print(f"\nTraining k-Means with {num_clusters} clusters...")

kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
kmeans.fit(latent_vectors_flat)

joblib.dump(kmeans, 'kmeans_codebook.joblib')

print("✅ k-Means training complete!")
print(f"Codebook created with {kmeans.cluster_centers_.shape[0]} vectors.")
