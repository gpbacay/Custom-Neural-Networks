# Spatial Graph Convolutional Network (SGCN)
Bacay, Gianne P. (2024)

## Table of Contents
1. [Introduction](#introduction)
2. [What is a Spatial Graph Convolutional Network (SGCN)?](#what-is-a-spatial-graph-convolutional-network-sgcn)
3. [Architecture of SGCN](#architecture-of-sgcn)
4. [How Does SGCN Work?](#how-does-sgcn-work)
5. [Implementation of SGCN](#implementation-of-sgcn)
   - [Import Libraries](#import-libraries)
   - [Define SGCN Layers](#define-sgcn-layers)
   - [Create Graph from Image](#create-graph-from-image)
   - [Preprocess MNIST Data](#preprocess-mnist-data)
   - [Train SGCN](#train-sgcn)
   - [Test SGCN](#test-sgcn)
   - [Main Function](#main-function)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [Advantages and Disadvantages of SGCN](#advantages-and-disadvantages-of-sgcn)
9. [Applications of SGCN](#applications-of-sgcn)

## Introduction
The Spatial Graph Convolutional Network (SGCN) is a specialized model designed for processing graph-structured data, particularly suited for image classification tasks. It integrates graph convolutional layers with spatial graph processing to enhance feature learning and classification performance on image datasets like MNIST.

## What is a Spatial Graph Convolutional Network (SGCN)?
The SGCN utilizes graph-based approaches to process image data, where each pixel in the image is considered a node in a graph. This network includes:

- Spatial Graph Convolutional Layers: These layers handle graph-structured data by aggregating and transforming features based on spatial relationships between nodes.
- Graph Creation from Images: Converts images into graph representations, with nodes corresponding to pixels and edges representing spatial connectivity.

## Architecture of SGCN

The architecture of SGCN consists of the following key components:

1. Input Layer:
   - Receives image data and converts it into a graph representation.
2. Spatial Graph Convolutional Layers:
   - Processes graph-structured data to extract features from spatially connected nodes.
3. Fully Connected Layer:
   - Aggregates features from graph convolutional layers to produce class predictions.

## How Does SGCN Work?
1. Graph Construction:
   - Converts image data into a graph with pixels as nodes and adjacency based on spatial proximity.
2. Feature Aggregation:
   - Uses Spatial Graph Convolutional Layers to aggregate and transform features based on the graph structure.
3. Classification:
   - Passes the aggregated features through a fully connected layer to classify the image into predefined categories.

## Implementation of SGCN
Here's a step-by-step guide to implementing SGCN using PyTorch:

### Import Libraries

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
```

### Define SGCN Layers

```python
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
```

### Create Graph from Image

```python
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
```

### Preprocess MNIST Data

```python
def preprocess_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    return train_dataset, test_dataset
```

### Train SGCN

```python
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
```

### Test SGCN

```python
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
```

### Main Function

```python
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
```

## Results
The SGCN achieved a test accuracy of 0.1128 on the MNIST dataset. This result indicates that while the model integrates graph-based spatial features, further tuning and model improvements may be required to enhance performance.

## Conclusion
The Spatial Graph Convolutional Network (SGCN) effectively utilizes graph-based representations for image classification. Its architecture, incorporating spatial graph convolutional layers, offers a novel approach to feature extraction from image data. The results suggest that while SGCN has potential, additional refinements and optimizations are necessary for achieving higher accuracy.

## Advantages and Disadvantages of SGCN
### Advantages
- Graph-Based Features: Leverages spatial relationships between pixels.
- Dynamic Feature Aggregation: Capable of capturing complex spatial features.

### Disadvantages
- Accuracy: Achieved lower accuracy compared to conventional models.
- Computational Complexity: Potentially higher complexity due to graph operations.

## Applications of SGCN
- Image Classification: Suitable for tasks requiring spatial feature extraction.
- Graph-Based Data Analysis: Applicable in domains where data can be represented as graphs.

