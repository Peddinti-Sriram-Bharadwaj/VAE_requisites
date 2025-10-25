import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from safetensors.torch import save_file

latent_dim = 16
batch_size = 128
learning_rate = 1e-3
num_epochs = 20

device = torch.device("mps")

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("Data loaded successfully")
print(f"Using device: {device}")


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten()
                )

        self.fc_mu = nn.Linear(32*7*7, latent_dim)
        self.fc_log_var = nn.Linear(32*7*7, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 32*7*7)
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(), 
                nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
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
        reconstructed = self.decoder(z)

        return reconstructed, mu, log_var

model = VAE().to(device)

print("VAE model created")
print(model)


def vae_loss_function(reconstructed_x, x, mu, log_var):
    recon_loss = nn.functional.binary_cross_entropy(reconstructed_x, x, reduction="sum")

    kl_divergence = -0.5 * torch.sum( 1 + log_var - mu.pow(2))

    return recon_loss + kl_divergence

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

print("Starting VAE training..")
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (data,_) in enumerate(train_loader):
        data = data.to(device)

        reconstructed, mu, log_var = model(data)

        loss = vae_loss_function(reconstructed, data, mu, log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss/ len(train_dataset)
    print(f"Epoch [{epoch + 1} / {num_epochs}], Average loss : {avg_loss:.4f}")

print("\n VAE training complete")

model_path = "vae_model.safetensors"
save_file(model.state_dict(), model_path)
print(f"VAE model saved to {model_path}")




