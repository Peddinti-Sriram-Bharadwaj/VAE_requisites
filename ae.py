import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from safetensors.torch import save_file

# --- Step 1: Data Loading (Unchanged) ---
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# --- Step 2: NEW Convolutional Autoencoder Model ---
class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        # Encoder Layers
        self.encoder = nn.Sequential(
            # Input: (batch_size, 1, 28, 28)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # -> (16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> (32, 7, 7)
            nn.ReLU()
        )
        
        # Decoder Layers
        self.decoder = nn.Sequential(
            # Input: (32, 7, 7)
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (1, 28, 28)
            nn.Sigmoid() # Squashes output to be between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Use the new model
# Since you're on a Mac with an M3 chip, we can use the 'mps' device for acceleration.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ConvolutionalAutoencoder().to(device)
print("Using new Convolutional Autoencoder on device:", device)

# --- Step 3: Train the Model ---
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 20 # A CNN learns faster, so 20 epochs should be plenty

loss_history = []
print("Starting training...")

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for data in train_loader:
        images, _ = data
        images = images.to(device)
        
        reconstructed = model(images)
        loss = criterion(reconstructed, images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

print("\n✅ Training complete!")

# --- Step 4: Save Model and Loss Plot ---
model_path = "autoencoder_model.safetensors"
# Note: The model's state_dict is now different, but the saving process is the same.
save_file(model.state_dict(), model_path)
print(f"✅ Model saved to {model_path}")

plt.figure()
plt.plot(loss_history)
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig("loss_plot.png")
print("✅ Loss plot saved to loss_plot.png")
