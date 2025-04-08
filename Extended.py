import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn import global_mean_pool, HypergraphConv
from torch_cluster import knn_graph, radius_graph
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree
from collections import defaultdict
from torch_geometric.nn import GCNConv, TransformerConv
from torch_geometric.nn import MessagePassing
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_to_point_cloud(image, z_value, threshold):
    indices = np.argwhere(image > threshold)
    intensities = image[indices[:, 0], indices[:, 1]]
    z_coords = np.full((indices.shape[0], 1), z_value)
    point_cloud = np.column_stack((indices, z_coords, intensities))
    return point_cloud

file_path = "dataset/quarkds.hdf5"

with h5py.File(file_path, "r") as f:
    dataset_size = f["X_jets"].shape[0]
    subset_size = 50000
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

# Normalize each channel (using the max over the subset)
def normalize_per_image(img):
    mean = np.mean(img)
    std = np.std(img) + 1e-9  # Avoid division by zero
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

def connected_components(n_nodes, edges):
    parent = list(range(n_nodes))
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_v] = root_u
    
    for u, v in edges:
        union(u, v)
    
    components = defaultdict(set)
    for u in range(n_nodes):
        components[find(u)].add(u)
    
    return list(components.values())

def _build_hyperedge_index(hyperedges):
    indices = []
    for hedge_idx, hedge in enumerate(hyperedges):
        indices.extend([[node, hedge_idx] for node in hedge])
    if not indices:  # Handle empty hyperedges
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(indices, dtype=torch.long).t().contiguous()

def point_cloud_to_graph(ecal, hcal, track, m0_value, pt_value, y_value, 
                         intensity_percent=5, radius1=1.5, radius2=3.0,
                         n_freqs=4, k=5):
    # Combine point clouds
    point_cloud = np.vstack((ecal, hcal, track))
    pos_np = point_cloud[:, :3]
    intensity_np = point_cloud[:, 3]
    
    # Convert to tensors
    pos = torch.tensor(pos_np, dtype=torch.float)
    intensity = torch.tensor(intensity_np, dtype=torch.float).unsqueeze(1)
    
    # Original ParticleNet features
    layer_id = pos[:, 2:3]
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
    
    track_density = compute_track_density(pos, layer_id)
    
    #Hyperedge Construction
    tree = cKDTree(pos_np)
    max_radius = max(radius1, radius2)
    pairs = tree.query_pairs(max_radius, output_type='ndarray')
    
    valid_pairs = []
    if pairs.size > 0:
        i_vals = intensity_np[pairs[:, 0]]
        j_vals = intensity_np[pairs[:, 1]]
        rel_diff = np.abs(i_vals - j_vals) / np.maximum(np.maximum(np.abs(i_vals), np.abs(j_vals)), 1e-6)
        spatial_mask = np.linalg.norm(pos_np[pairs[:, 0]] - pos_np[pairs[:, 1]], axis=1) <= radius1
        intensity_mask = rel_diff <= (intensity_percent/100)
        valid_pairs = pairs[np.logical_or(spatial_mask, intensity_mask)]
    
    hyperedges = connected_components(pos_np.shape[0], valid_pairs)
    
    hyperedge_features = []
    for hedge in hyperedges:
        nodes = list(hedge)
        if len(nodes) > 1:
            diffs = pos_np[nodes][:, None] - pos_np[nodes]
            dists = np.sqrt((diffs**2).sum(axis=2))
            avg_dist = np.triu(dists, 1).sum() / (len(nodes)*(len(nodes)-1)/2)
            max_diff = intensity_np[nodes].max() - intensity_np[nodes].min()
        else:
            avg_dist = 0.0
            max_diff = 0.0
        hyperedge_features.append([avg_dist, max_diff])
    
    hyperedge_index = _build_hyperedge_index(hyperedges)
    if hyperedge_index.size(1) == 0:
        dummy_node = 0 if len(pos_np) > 0 else 0
        hyperedge_index = torch.tensor([[dummy_node], [0]], dtype=torch.long)
        hyperedge_features = [[0.0, 0.0]]
    
    _, knn_indices = tree.query(pos_np, k=k+1)
    knn_indices = knn_indices[:, 1:].reshape(-1)
    row_indices = np.repeat(np.arange(len(pos_np)), k)
    knn_pairs = np.column_stack([row_indices, knn_indices])
    
    node_hedge_map = defaultdict(set)
    for hedge_idx, hedge in enumerate(hyperedges):
        for node in hedge:
            node_hedge_map[node].add(hedge_idx)
            
    mask = []
    for i, j in knn_pairs:
        mask.append(len(node_hedge_map[i] & node_hedge_map[j]) > 0)
    filtered_edges = knn_pairs[np.array(mask, dtype=bool)]
    
    edge_index = torch.tensor(filtered_edges.T, dtype=torch.long)
    
    node_degrees = torch.zeros(pos.shape[0], 1)
    for hedge in hyperedges:
        nodes = list(hedge)
        node_degrees[nodes] += 1
    is_isolated = (node_degrees == 0).float() * 10.0
    orig_features = torch.cat([node_degrees, intensity, layer_id, track_density, is_isolated], dim=1)
    
    freqs = 2 ** np.arange(n_freqs)
    pos_expanded = pos_np[:, :, np.newaxis]
    freqs_expanded = freqs[np.newaxis, np.newaxis, :]
    pos_embedding = np.sin(pos_expanded * freqs_expanded).reshape(len(pos_np), -1)
    extra_features = np.hstack([ intensity_np.reshape(-1, 1), pos_np[:, 2:].reshape(-1, 1), pos_embedding ])
    extra_features = torch.tensor(extra_features, dtype=torch.float32)
    
    x = torch.cat([orig_features, extra_features], dim=1)
    
    return Data(
        x=x,
        edge_index=edge_index,
        hyperedge_index=hyperedge_index,
        hyperedge_attr=torch.tensor(hyperedge_features, dtype=torch.float32),
        pos=pos,
        m0=torch.tensor([[m0_value]], dtype=torch.float),
        pt=torch.tensor([[pt_value]], dtype=torch.float),
        y=torch.tensor([y_value], dtype=torch.float)
    )

graph_data = [point_cloud_to_graph(e, h, t, m, p, label) 
              for e, h, t, m, p, label in zip(ecal_clouds, hcal_clouds, track_clouds, subset_m0, subset_pt, subset_y)]
print("Graphs created")

train_val_data, test_data = train_test_split(graph_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class HyperParticleNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.hyper_conv = HypergraphConv(in_channels, out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )
        self.residual = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels)

    def forward(self, x, hyperedge_index):
        out = self.hyper_conv(x, hyperedge_index)
        out = self.mlp(out)
        res = self.residual(x)
        return F.leaky_relu(out + res)

class HyperGraphMessage(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(HyperGraphMessage, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.attn_lin = nn.Linear(2 * out_channels, 1)
    def forward(self, x, hyperedge_index):
        x_trans = self.lin(x)
        return self.propagate(hyperedge_index, x=x_trans)
    def message(self, x_i, x_j):
        attn_input = torch.cat([x_i, x_j], dim=-1)
        attn_coeff = torch.sigmoid(self.attn_lin(attn_input))
        return attn_coeff * x_j
    def update(self, aggr_out):
        return aggr_out

class KNNGraphMessage(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(KNNGraphMessage, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.attn_lin = nn.Linear(2 * out_channels, 1)
    def forward(self, x, edge_index):
        x_trans = self.lin(x)
        return self.propagate(edge_index, x=x_trans)
    def message(self, x_i, x_j):
        attn_input = torch.cat([x_i, x_j], dim=-1)
        attn_coeff = torch.sigmoid(self.attn_lin(attn_input))
        return attn_coeff * x_j
    def update(self, aggr_out):
        return aggr_out

class CrossGraphAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossGraphAttention, self).__init__()
        self.hyper_msg = HyperGraphMessage(in_channels, out_channels)
        self.knn_msg = KNNGraphMessage(in_channels, out_channels)
        self.gate_lin = nn.Linear(2 * out_channels, 2)
    def forward(self, x, hyperedge_index, knn_edge_index):
        hyper_out = self.hyper_msg(x, hyperedge_index)
        knn_out = self.knn_msg(x, knn_edge_index)
        combined = torch.cat([hyper_out, knn_out], dim=-1)
        gate = torch.sigmoid(self.gate_lin(combined))
        gate_hyper = gate[:, 0].unsqueeze(-1)
        gate_knn = gate[:, 1].unsqueeze(-1)
        out = gate_hyper * hyper_out + gate_knn * knn_out
        return out

class ExtendedHyperParticleNetWithCrossAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, out_dim=2):
        super().__init__()
        self.pre_local = GCNConv(in_channels, in_channels)  # Local layer on raw features
        self.block1 = HyperParticleNetBlock(in_channels, hidden_dim)
        self.block2 = HyperParticleNetBlock(hidden_dim, hidden_dim * 2)
        self.block3 = HyperParticleNetBlock(hidden_dim * 2, hidden_dim * 4)
        self.cross_attn = CrossGraphAttention(hidden_dim * 4, hidden_dim * 4)
        self.local_conv = GCNConv(hidden_dim * 4, hidden_dim * 4)
        self.transformer_conv = TransformerConv(hidden_dim * 4, hidden_dim * 4, heads=4, concat=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, data):
        x_pre = self.pre_local(data.x, data.edge_index)  # Pre-local processing
        x = x_pre
        x = self.block1(x, data.hyperedge_index)
        x = self.block2(x, data.hyperedge_index)
        x = self.block3(x, data.hyperedge_index)
        x = self.cross_attn(x, data.hyperedge_index, data.edge_index) + x  # Residual from hyperblock output
        x_local = self.local_conv(x, data.edge_index)
        x = self.transformer_conv(x_local, data.edge_index) + x_local       # Residual in local conv
        x_pool = global_mean_pool(x, data.batch)
        x = torch.cat([x_pool, data.m0, data.pt], dim=1)
        return self.fc(x)


in_channels = 19
model = ExtendedHyperParticleNetWithCrossAttention(in_channels=in_channels, hidden_dim=64, out_dim=2).to(device)
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
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y.long())
            total_loss += loss.item() * data.num_graphs
            pred = output.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.y.size(0)
            probs = torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
            labels = data.y.detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)
    avg_loss = total_loss / total
    accuracy = correct / total * 100
    roc_auc = roc_auc_score(all_labels, all_probs)
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, ROC-AUC: {roc_auc:.4f}")
    return avg_loss


for epoch in range(20):
    print("Training!")
    train_loss = train()
    val_loss = validate()
    scheduler.step(val_loss)
    print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
print("TESTING!")
test_loss = test()
print(f"Test Loss: {test_loss:.4f}")
