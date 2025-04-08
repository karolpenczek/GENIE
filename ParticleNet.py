import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn import global_mean_pool, EdgeConv
from torch_cluster import knn_graph, radius_graph
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
subset_size = 50000

def image_to_point_cloud(image, z_value, threshold):
    indices = np.argwhere(image > threshold)
    intensities = image[indices[:, 0], indices[:, 1]]
    z_coords = np.full((indices.shape[0], 1), z_value)
    point_cloud = np.column_stack((indices, z_coords, intensities))
    return point_cloud

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

batch_size = 32
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

m0 = (subset_m0 - subset_m0.mean()) / subset_m0.std()
subset_pt = (subset_pt - subset_pt.mean()) / subset_pt.std()

def process_images_to_point_clouds(ecal_images, hcal_images, track_images):
    ecal_clouds = [image_to_point_cloud(img, z_value=0, threshold=1e-8) for img in ecal_images]
    hcal_clouds = [image_to_point_cloud(img, z_value=1, threshold=1e-4) for img in hcal_images]
    track_clouds = [image_to_point_cloud(img, z_value=2, threshold=1e-6) for img in track_images]
    return ecal_clouds, hcal_clouds, track_clouds

ecal_clouds, hcal_clouds, track_clouds = process_images_to_point_clouds(ECAL, HCAL, Tracks)

def compute_track_density(pos, layer_id, radius=2.0):
    track_mask = (layer_id.squeeze() == 2)
    track_positions = pos[track_mask]
    density_values = torch.zeros(pos.shape[0], dtype=torch.float)
    if track_positions.shape[0] > 0:
        tree = cKDTree(pos.numpy())
        for i in range(pos.shape[0]):
            neighbors = tree.query_ball_point(pos[i].numpy(), r=radius)
            track_neighbors = sum(track_mask[neighbors].numpy())
            density_values[i] = track_neighbors / max(len(neighbors), 1)
    return density_values.view(-1, 1)

def point_cloud_to_graph(ecal, hcal, track, m0_value, pt_value, y_value, scaling_factor=100):
    point_cloud = np.vstack((ecal, hcal, track))
    pos = torch.tensor(point_cloud[:, :3], dtype=torch.float)
    intensity = torch.tensor(point_cloud[:, 3:4], dtype=torch.float)
    layer_id = torch.tensor(point_cloud[:, 2:3], dtype=torch.float)
    track_density = compute_track_density(pos, layer_id)
    edge_index = radius_graph(pos, r=3)
    row, col = edge_index
    edge_distances = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)
    edge_similarities = torch.exp(-torch.abs(intensity[row] - intensity[col]) * scaling_factor)
    edge_attr = torch.cat((edge_distances, edge_similarities), dim=1)
    node_degrees = torch.bincount(edge_index[0], minlength=pos.shape[0]).float().view(-1, 1)
    x = torch.cat((node_degrees, intensity, layer_id, track_density), dim=1)
    is_isolated = (node_degrees == 0).float() * 10.0
    x = torch.cat((x, is_isolated), dim=1)
    m0_tensor = torch.tensor([m0_value], dtype=torch.float)
    pt_tensor = torch.tensor([pt_value], dtype=torch.float)
    y_tensor = torch.tensor([y_value], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, m0=m0_tensor, pt=pt_tensor, y=y_tensor)

graph_data = [point_cloud_to_graph(e, h, t, m, p, label) for e, h, t, m, p, label in zip(ecal_clouds, hcal_clouds, track_clouds, subset_m0, subset_pt, subset_y)]
print("Graphs created")
train_val_data, test_data = train_test_split(graph_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class ParticleNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ParticleNetBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.edge_conv = EdgeConv(self.mlp, aggr='max')
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(in_channels, out_channels)
    def forward(self, x, edge_index):
        out = self.edge_conv(x, edge_index)
        res = self.residual(x)
        return F.leaky_relu(out + res)

class ParticleNet(nn.Module):
    def __init__(self, in_channels=5):
        super(ParticleNet, self).__init__()
        self.block1 = ParticleNetBlock(in_channels, 16)
        self.block2 = ParticleNetBlock(16, 32)
        self.block3 = ParticleNetBlock(32, 128)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 2)
    def forward(self, data):
        x, pos, batch, edge_index = data.x, data.pos, data.batch, data.edge_index
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = self.block3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc2(x)
        return x

model = ParticleNet(in_channels=5).to(device)
print("model loaded")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        pred = output.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.y.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total * 100
    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss

def validate():
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y.long())
            total_loss += loss.item() * data.num_graphs
            pred = output.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.y.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total * 100
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss

def test():
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y.long())
            total_loss += loss.item() * data.num_graphs
            probs = F.softmax(output, dim=1)
            all_preds.append(probs.cpu())
            all_labels.append(data.y.cpu())
            pred = output.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.y.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total * 100
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    roc_auc = roc_auc_score(all_labels, all_preds[:, 1])
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, ROC-AUC: {roc_auc:.4f}")
    return avg_loss

print("Training!")
for epoch in range(10):
    train_loss = train()
    val_loss = validate()
    scheduler.step(val_loss)
    print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
print("TESTING!")
test_loss = test()
print(f"Test Loss: {test_loss:.4f}")
