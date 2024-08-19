import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Sample SQL data for demonstration
# Define your tables and relationships here
data = {
    'users': [
        {'user_id': 1, 'name': 'Alice'},
        {'user_id': 2, 'name': 'Bob'}
    ],
    'purchases': [
        {'purchase_id': 1, 'user_id': 1, 'amount': 100},
        {'purchase_id': 2, 'user_id': 2, 'amount': 150}
    ]
}

# Create a graph
G = nx.Graph()

# Add nodes for users
for user in data['users']:
    G.add_node(user['user_id'], type='user', name=user['name'])

# Add nodes for purchases
for purchase in data['purchases']:
    G.add_node(purchase['purchase_id'], type='purchase', amount=purchase['amount'])
    # Add edges between users and purchases
    G.add_edge(purchase['user_id'], purchase['purchase_id'])

# Convert graph to PyTorch Geometric Data object
def graph_to_pyg_data(G):
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    num_nodes = len(G.nodes)
    x = torch.eye(num_nodes, dtype=torch.float)  # One-hot encoding for node features
    data = Data(x=x, edge_index=edge_index)
    return data

data = graph_to_pyg_data(G)

# Define a simple GNN model
class GNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 32)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize and train the model
model = GNN(num_node_features=data.num_node_features)

# Dummy training loop (for illustration purposes only)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# For demonstration, we use random target embeddings
target_embeddings = torch.randn(data.num_nodes, 32)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out, target_embeddings)
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(100):
    loss = train()
    print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')

# Generate node embeddings
model.eval()
with torch.no_grad():
    node_embeddings = model(data)

# Print node embeddings
print("Node Embeddings:")
for i, node_id in enumerate(G.nodes):
    print(f'Node {node_id}: {node_embeddings[i].numpy()}')

# Graph Neural Network (GNN)
# python gnn_sql_to_emb_v1.py
