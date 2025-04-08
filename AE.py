import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# ------------------------------
# Data loading and preprocessing
# ------------------------------
subset_size = 15000
file_path = "dataset/quarkds.hdf5"
with h5py.File(file_path, "r") as f:
    dataset_size = f["X_jets"].shape[0]
    subset_indices = np.random.choice(dataset_size, subset_size, replace=False)
    subset_indices.sort()
    X_jets = np.array(f["X_jets"][subset_indices])
    m0 = np.array(f["m0"][subset_indices])
    pt = np.array(f["pt"][subset_indices])
    y = np.array(f["y"][subset_indices])

X_jets = torch.tensor(X_jets, dtype=torch.float32)
m0 = torch.tensor(m0, dtype=torch.float32)
pt = torch.tensor(pt, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

class JetDataset(Dataset):
    def __init__(self, X_jets, m0, pt, y):
        self.X_jets = X_jets
        self.m0 = m0
        self.pt = pt
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        image = self.X_jets[idx]
        m0_value = self.m0[idx]
        pt_value = self.pt[idx]
        label = self.y[idx]
        return image, m0_value, pt_value, label

dataset = JetDataset(X_jets, m0, pt, y)
subset_indices = np.random.choice(len(dataset), subset_size, replace=False)
subset_data = [dataset[i] for i in subset_indices]
subset_images = np.stack([item[0].numpy() for item in subset_data])
subset_m0 = np.array([item[1].item() for item in subset_data])
subset_pt = np.array([item[2].item() for item in subset_data])
subset_y = np.array([item[3].item() for item in subset_data])
ECAL = subset_images[:, :, :, 0]
HCAL = subset_images[:, :, :, 1]
Tracks = subset_images[:, :, :, 2]

def normalize_per_image(img):
    mean = np.mean(img)
    std = np.std(img) + 1e-9
    return (img - mean) / std

ECAL = np.array([normalize_per_image(img) for img in ECAL])
HCAL = np.array([normalize_per_image(img) for img in HCAL])
Tracks = np.array([normalize_per_image(img) for img in Tracks])

# ------------------------------
# Define the Autoencoder Model
# ------------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder_ecal = self._build_encoder()
        self.encoder_hcal = self._build_encoder()
        self.encoder_tracks = self._build_encoder()

        self.fc_latent = nn.Linear(43200, 256)  # Concatenated size from flattened encoder outputs
        self.fc_decode = nn.Linear(256, 43200)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64*3, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output 3 channels corresponding to ECAL, HCAL, and Tracks
        )

    def _build_encoder(self):
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, ecal, hcal, tracks):
        ecal_latent = self.encoder_ecal(ecal)
        hcal_latent = self.encoder_hcal(hcal)
        tracks_latent = self.encoder_tracks(tracks)

        ecal_flat = ecal_latent.view(ecal_latent.size(0), -1)
        hcal_flat = hcal_latent.view(hcal_latent.size(0), -1)
        tracks_flat = tracks_latent.view(tracks_latent.size(0), -1)

        combined_latent = torch.cat([ecal_flat, hcal_flat, tracks_flat], dim=1)

        latent_vector = self.fc_latent(combined_latent)

        decoded_fc = self.fc_decode(latent_vector)
        decoded_fc = decoded_fc.view(decoded_fc.size(0), 192, 15, 15)

        reconstructed = self.decoder(decoded_fc)
        return reconstructed

# ------------------------------
# Model, Optimizer, Loss
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ------------------------------
# Data splitting and DataLoaders
# ------------------------------
train_ratio = 0.8
train_size = int(train_ratio * subset_size)
test_size = subset_size - train_size

train_indices, test_indices = train_test_split(np.arange(subset_size), test_size=test_size, random_state=42)

# Create torch tensors for each channel and move to device
ecal_train   = torch.tensor(ECAL[train_indices], dtype=torch.float32).unsqueeze(1).to(device)
hcal_train   = torch.tensor(HCAL[train_indices], dtype=torch.float32).unsqueeze(1).to(device)
tracks_train = torch.tensor(Tracks[train_indices], dtype=torch.float32).unsqueeze(1).to(device)

ecal_test   = torch.tensor(ECAL[test_indices], dtype=torch.float32).unsqueeze(1).to(device)
hcal_test   = torch.tensor(HCAL[test_indices], dtype=torch.float32).unsqueeze(1).to(device)
tracks_test = torch.tensor(Tracks[test_indices], dtype=torch.float32).unsqueeze(1).to(device)

# The target is the concatenation of the three channels
target_train = torch.cat([ecal_train, hcal_train, tracks_train], dim=1)
target_test  = torch.cat([ecal_test, hcal_test, tracks_test], dim=1)

train_dataset = TensorDataset(ecal_train, hcal_train, tracks_train, target_train)
test_dataset  = TensorDataset(ecal_test, hcal_test, tracks_test, target_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ------------------------------
# Training Loop
# ------------------------------
for epoch in range(10):
    model.train()
    running_loss = 0.0
    
    for ecal_batch, hcal_batch, tracks_batch, target_batch in train_loader:
        ecal_batch = ecal_batch.to(device)
        hcal_batch = hcal_batch.to(device)
        tracks_batch = tracks_batch.to(device)
        target_batch = target_batch.to(device)
        
        optimizer.zero_grad()
        
        output = model(ecal_batch, hcal_batch, tracks_batch)
        
        # Adjust output size if necessary via interpolation
        if output.shape[-2:] != target_batch.shape[-2:]:
            output = F.interpolate(output, size=target_batch.shape[-2:], mode='bilinear', align_corners=False)
        
        rec_loss = criterion(output, target_batch)
        loss = rec_loss

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("Time: ", current_time)
    print("Rec loss: ", rec_loss.item())
    print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader):.4f}")

# ------------------------------
# Testing and Evaluation
# ------------------------------
model.eval()
test_loss = 0.0

with torch.no_grad():
    for ecal_batch, hcal_batch, tracks_batch, target_batch in test_loader:
        ecal_batch = ecal_batch.to(device)
        hcal_batch = hcal_batch.to(device)
        tracks_batch = tracks_batch.to(device)
        target_batch = target_batch.to(device)

        output = model(ecal_batch, hcal_batch, tracks_batch)
        output = F.interpolate(output, size=target_batch.shape[-2:], mode='bilinear', align_corners=False)

        rec_loss = criterion(output, target_batch)
        test_loss += rec_loss.item()
print("Testing!")
print("Rec loss (last batch): ", rec_loss.item())
print(f"Test Loss: {test_loss / len(test_loader):.4f}")

# ------------------------------
# Plotting Original and Reconstructed Images
# ------------------------------
# We'll select one batch from the test_loader and plot three samples from that batch.
# For each sample we'll display the original channel image and the reconstructed one.

# Get one batch from test_loader
with torch.no_grad():
    sample_batch = next(iter(test_loader))
    ecal_sample, hcal_sample, tracks_sample, target_sample = sample_batch
    ecal_sample = ecal_sample.to(device)
    hcal_sample = hcal_sample.to(device)
    tracks_sample = tracks_sample.to(device)
    target_sample = target_sample.to(device)
    reconstructed_sample = model(ecal_sample, hcal_sample, tracks_sample)
    # Resize reconstructed output if needed
    reconstructed_sample = F.interpolate(reconstructed_sample, size=target_sample.shape[-2:], mode='bilinear', align_corners=False)

# Move to CPU and convert to NumPy arrays for plotting
target_sample_np = target_sample.cpu().numpy()    # shape: [batch, 3, H, W]
reconstructed_np = reconstructed_sample.cpu().numpy()

# Plot 3 samples from the batch
channel_names = ["ECAL", "HCAL", "Tracks"]
num_samples_to_plot = 3

for i in range(num_samples_to_plot):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    # For original target, channels are ordered as [0: ECAL, 1: HCAL, 2: Tracks]
    for j in range(3):
        # Original image for channel j
        axs[0, j].imshow(target_sample_np[i, j], cmap="viridis")
        axs[0, j].set_title(f"Original {channel_names[j]}")
        axs[0, j].axis("off")
        
        # Reconstructed image for channel j
        axs[1, j].imshow(reconstructed_np[i, j], cmap="viridis")
        axs[1, j].set_title(f"Reconstructed {channel_names[j]}")
        axs[1, j].axis("off")
    plt.tight_layout()
    plt.show()
