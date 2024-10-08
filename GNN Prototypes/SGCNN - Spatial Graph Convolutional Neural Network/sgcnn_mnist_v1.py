import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

class SpatialGraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SpatialGraphConvLayer, self).__init__()
        # Initialize weight matrix with correct dimensions
        self.U = nn.Parameter(torch.randn(2, out_features))  # Adjust to (2, out_features)
        self.b = nn.Parameter(torch.randn(out_features))
        
    def forward(self, x, adj, coords):
        num_nodes = x.size(0)
        out_features = self.U.size(1)
        
        # Initialize output feature matrix
        aggregated_features = torch.zeros(num_nodes, out_features).to(x.device)
        
        for i in range(num_nodes):
            neighbors = (adj[i] > 0).nonzero(as_tuple=True)[0]
            if neighbors.numel() > 0:
                diff = coords[neighbors] - coords[i]
                diff = diff @ self.U  # Ensure dimensions align
                diff = F.relu(diff + self.b)
                aggregated_features[i] = torch.mean(diff, dim=0)
                
        return aggregated_features

class SGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(SGCN, self).__init__()
        self.layer1 = SpatialGraphConvLayer(in_features, hidden_features)
        self.layer2 = SpatialGraphConvLayer(hidden_features, out_features)
        self.fc = nn.Linear(out_features, out_features)

    def forward(self, x, adj, coords):
        x = self.layer1(x, adj, coords)
        x = F.relu(x)
        x = self.layer2(x, adj, coords)
        x = self.fc(x)
        return x

# Create a sample graph
def create_sample_graph(num_nodes=10):
    G = nx.erdos_renyi_graph(num_nodes, 0.5)
    adj = nx.to_numpy_array(G)
    adj = torch.tensor(adj, dtype=torch.float32)
    
    # Node features and spatial coordinates
    node_features = torch.randn(num_nodes, 5)
    spatial_coords = torch.randn(num_nodes, 2)  # Example 2D coordinates
    
    return node_features, adj, spatial_coords

# Initialize and run the model
def main():
    num_nodes = 10
    node_features, adj, spatial_coords = create_sample_graph(num_nodes)
    
    model = SGCN(in_features=5, hidden_features=10, out_features=3)
    output = model(node_features, adj, spatial_coords)
    print("Model output:")
    print(output)

if __name__ == "__main__":
    main()


# Spatial Graph Convolutional Neural Network (SGCNN)
# python sgcnn_mnist_v1.py
"""
Model output:
tensor([[ 0.1383, -0.2716, -0.5308],
        [-1.1391,  0.5082, -1.8163],
        [-0.0490,  0.1681, -1.0342],
        [ 0.8094,  1.0198, -1.5115],
        [-0.2347,  0.0437, -0.9922],
        [-0.2302, -0.2120, -0.7503],
        [ 0.6872,  1.1230, -1.6592],
        [ 0.4757,  0.6128, -1.2588],
        [ 0.5154,  0.5065, -1.1349],
        [ 0.2740, -0.1028, -0.6411]], grad_fn=<AddmmBackward0>)
        
"""
