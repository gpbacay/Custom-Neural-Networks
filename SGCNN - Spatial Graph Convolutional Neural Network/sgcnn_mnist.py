import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

class SGCN(nn.Module):
    class SpatialGraphConvLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super(SGCN.SpatialGraphConvLayer, self).__init__()
            self.U = nn.Parameter(torch.randn(2, out_features))
            self.b = nn.Parameter(torch.randn(out_features))
        
        def forward(self, x, adj, coords):
            num_nodes = x.size(0)
            out_features = self.U.size(1)
            
            aggregated_features = torch.zeros(num_nodes, out_features).to(x.device)
            
            for i in range(num_nodes):
                neighbors = (adj[i] > 0).nonzero(as_tuple=True)[0]
                if neighbors.numel() > 0:
                    diff = coords[neighbors] - coords[i]
                    diff = diff @ self.U
                    diff = F.relu(diff + self.b)
                    aggregated_features[i] = torch.mean(diff, dim=0)
            
            return aggregated_features

    def __init__(self, in_features, hidden_features, out_features):
        super(SGCN, self).__init__()
        self.layer1 = self.SpatialGraphConvLayer(in_features, hidden_features)
        self.layer2 = self.SpatialGraphConvLayer(hidden_features, out_features)
        self.fc = nn.Linear(out_features, 10)  # 10 classes for MNIST

    def forward(self, x, adj, coords):
        x = self.layer1(x, adj, coords)
        x = F.relu(x)
        x = self.layer2(x, adj, coords)
        x = self.fc(x)
        return x

def create_graph_from_image(image):
    num_nodes = image.size(0)
    adj = np.zeros((num_nodes, num_nodes))
    spatial_coords = np.array([(i // 28, i % 28) for i in range(num_nodes)])
    
    # Define adjacency based on pixel proximity (4-connectivity)
    for i in range(28):
        for j in range(28):
            index = i * 28 + j
            if i > 0:
                adj[index, index - 28] = 1
            if j > 0:
                adj[index, index - 1] = 1
            if i < 27:
                adj[index, index + 28] = 1
            if j < 27:
                adj[index, index + 1] = 1
    
    adj = torch.tensor(adj, dtype=torch.float32)
    spatial_coords = torch.tensor(spatial_coords, dtype=torch.float32)
    
    return adj, spatial_coords

def preprocess_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    return train_dataset, test_dataset

def train_sgcn(model, dataloader, optimizer, criterion, device):
    model.train()
    for images, labels in dataloader:
        batch_size = images.size(0)
        node_features = images.view(batch_size, -1)  # Flatten images to 784 features
        adj, spatial_coords = create_graph_from_image(images[0].view(-1))
        adj = adj.to(device)
        spatial_coords = spatial_coords.to(device)
        node_features = node_features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = model(node_features, adj, spatial_coords)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

def test_sgcn(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            batch_size = images.size(0)
            node_features = images.view(batch_size, -1)  # Flatten images to 784 features
            adj, spatial_coords = create_graph_from_image(images[0].view(-1))
            adj = adj.to(device)
            spatial_coords = spatial_coords.to(device)
            node_features = node_features.to(device)
            labels = labels.to(device)
            
            outputs = model(node_features, adj, spatial_coords)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SGCN(in_features=784, hidden_features=64, out_features=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset, test_dataset = preprocess_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Train the model
    train_sgcn(model, train_dataloader, optimizer, criterion, device)
    
    # Test the model
    test_accuracy = test_sgcn(model, test_dataloader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()




# Spatial Graph Convolutional Neural Network (SGCNN)
# python sgcnn_mnist.py
# Test Accuracy: 0.1128

