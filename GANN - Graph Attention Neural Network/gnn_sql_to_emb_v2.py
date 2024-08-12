import re
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

SQL_FILE_PATH = 'portfolio_db.sql'

def preprocess_sql_file(sql_file_path):
    """Preprocesses the SQL file to extract table data."""
    with open(sql_file_path, 'r') as file:
        sql_script = file.read()
    
    sql_script = re.sub(r'--.*', '', sql_script)  # Remove comments
    sql_script = re.sub(r'\s+', ' ', sql_script)  # Normalize whitespace

    statements = re.split(r';\s*', sql_script)
    return [s.strip() for s in statements if s.strip()]

def parse_statements(statements):
    """Parse SQL statements to extract table data."""
    tables = {}
    for statement in statements:
        if statement.upper().startswith('CREATE TABLE'):
            match = re.match(r'CREATE TABLE (\w+) \((.*)\)', statement, re.IGNORECASE | re.DOTALL)
            if match:
                table_name = match.group(1)
                columns = re.findall(r'(\w+)\s+\w+(?:\([^)]+\))?', match.group(2))
                tables[table_name] = {'columns': columns, 'data': []}
        elif statement.upper().startswith('INSERT INTO'):
            match = re.match(r'INSERT INTO (\w+) \((.*?)\) VALUES (.*)', statement, re.IGNORECASE | re.DOTALL)
            if match:
                table_name = match.group(1)
                columns = [col.strip() for col in match.group(2).split(',')]
                values_str = match.group(3)
                values = re.findall(r'\((.*?)\)', values_str)
                for value in values:
                    row = [v.strip().strip("'") for v in value.split(',')]
                    tables[table_name]['data'].append(row)
    return tables

def build_feature_matrix(tables):
    """Build a feature matrix from table data."""
    all_data = []
    for table_name, table_info in tables.items():
        columns = table_info['columns']
        data = table_info['data']
        for row in data:
            features = [float(x) if x.replace('.', '', 1).isdigit() else 0.0 for x in row]
            all_data.append(features)
    if not all_data:
        return None
    return np.array(all_data)

class SimpleNN(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_model(features, epochs=10, batch_size=32):
    """Train a simple neural network to generate embeddings."""
    input_dim = features.shape[1]
    embedding_dim = 32
    
    model = SimpleNN(input_dim, embedding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for batch_features, in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, outputs)  # Dummy target for self-supervised training
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model

def generate_embeddings(model, features):
    """Generate embeddings using the trained model."""
    model.eval()
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32)
        embeddings = model(features_tensor)
    return embeddings.numpy()

def save_embeddings(embeddings, output_dir, file_prefix='embedding_'):
    """Save embeddings to the specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, embedding in enumerate(embeddings):
        file_path = os.path.join(output_dir, f"{file_prefix}{i}.npy")
        np.save(file_path, embedding)
        print(f"Saved embedding to {file_path}")

# Main process
statements = preprocess_sql_file(SQL_FILE_PATH)
tables = parse_statements(statements)
features = build_feature_matrix(tables)

if features is not None and len(features) > 0:
    model = train_model(features)
    embeddings = generate_embeddings(model, features)
    save_embeddings(embeddings, 'Embeddings')
else:
    print("No valid features extracted. Please check your SQL data.")



# Graph Neural Network (GNN)
# python gnn_sql_to_emb_v2.py
# Remarks: Error