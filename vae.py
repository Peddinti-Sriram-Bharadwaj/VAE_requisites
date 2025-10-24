import torch
import torch.nn as nn
from torchvision import datasets , transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(root = './data', train=True, download = True, transform = transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Data loaded successfully")
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of test batches: {len(test_loader)}")

input_dim = 784

latent_dim = 32

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256), 
                nn.ReLU(), 
                nn.Linear(256, 128), 
                nn.ReLU(), 
                nn.Linear(128, latent_dim)
                )

        self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128), 
                nn.ReLU(), 
                nn.Linear(128, 256), 
                nn.ReLU(), 
                nn.Linear(256, input_dim), 
                nn.Sigmoid()
                )

    def forward(self, x):
        x = x.view(x.size(0), -1)

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        decoded = decoded.view(x.size(0), 1, 28, 28)

        return decoded

model = Autoencoder()

device = torch.device(
        "mps")
model.to(device)
print("Model created and moved to device: " , device)
print(model)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10

for epoch in range(num_epochs):
    for data in train_loader:
        images, _ = data
        images = images.to(device)

        reconstructed = model(images)

        loss = criterion(reconstructed, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
print("\n Training complete!")


