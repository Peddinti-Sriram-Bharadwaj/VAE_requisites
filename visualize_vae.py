import torch
import torch.nn as nn
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from safetensors.torch import load_file
import umap.umap_ as umap 
import numpy as np

latent_dim = 16

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(), 
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(), 
                nn.Flatten()
        )

        self.fc_mu = nn.Linear(32*7*7, latent_dim)
        self.fc_log_var = nn.Linear(32*7*7, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 32*7*7)
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(), 
                nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
                )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        z = self.reparameterize(mu, log_var)
        z = self.decoder_input(z)
        z = z.view(-1, 32, 7, 7)
        reconstructed  = self.decoder(z)
        return reconstructed, mu, log_var

device = torch.device("mps")
model = VAE()
state_dict = load_file("vae_model.safetensors", device = str(device))
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("VAE model loaded successfuly")

transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

all_latents = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        _, mu, _ = model(images)
        all_latents.append(mu.cpu().numpy())
        all_labels.append(labels.cpu().numpy())


all_latents = np.concatenate(all_latents, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
print(f"Generated latent vectors for {len(all_labels)} test images.")

print("Running UMAP to create 2d projection...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(all_latents)
print("UMAP complete")

# Create the plot
plt.figure(figsize=(12, 10))
# Scatter plot, colored by the digit labels
scatter = plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=all_labels,
    cmap='Spectral', # A good colormap for categorical data
    s=5 # Size of the points
)
plt.title('2D UMAP Projection of MNIST Latent Space', fontsize=18)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.colorbar(scatter, ticks=range(10), label='Digit Label')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()



