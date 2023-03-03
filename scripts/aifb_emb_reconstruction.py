import torch
import torch.nn as nn
from torch_geometric.data import Data

import pandas as pd
import numpy as np

class EmbeddingModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(EmbeddingModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.edge_weights = nn.Parameter(torch.Tensor([0.5, 0.5])) # Initialize edge weights with equal importance
        
        self.transform_layer = nn.Sequential(
            nn.Linear(embedding_dim*2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, edges): #dense neural network, gcnn(rgcn), r-gan 
        src_nodes = edges.src['node']
        dst_nodes = edges.dst['node']
        src_embed = self.node_embeddings(src_nodes)
        dst_embed = self.node_embeddings(dst_nodes)
        
        # Compute edge weights based on the edge type
        edge_type = edges.data['type']
        edge_weights = torch.index_select(self.edge_weights, 0, edge_type)
        
        # Compute the weighted sum of neighboring node embeddings
        weighted_sum = edge_weights[0]*src_embed + edge_weights[1]*dst_embed
        neighbors_embed = torch.zeros_like(src_embed)
        neighbors_embed.index_add_(0, dst_nodes, weighted_sum)
        neighbors_embed.index_add_(0, src_nodes, weighted_sum)
        
        # Concatenate the original node embeddings and neighboring node embeddings
        node_embed = torch.cat([self.node_embeddings.weight, neighbors_embed], dim=1)
        
        # Apply a transformation layer to compute the dynamic embeddings
        dynamic_embed = self.transform_layer(node_embed)
        
        return dynamic_embed
    

    
if __name__ == '__main__':
    homedir = "C:/Users/luisa/Projekte/Masterthesis/AIFB"
    df = pd.read_csv(homedir + "/data/aifb_without_literals.tsv", sep="\t")

    src_nodes = df['subject'].tolist()
    dst_nodes = df['object'].tolist()
    edge_types = df['predicate'].tolist()
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    data = Data(edge_index=edge_index, edge_type=edge_type)

    model = EmbeddingModel(num_node_features=0, embedding_dim=128, num_edge_types=len(edge_types), num_layers=3)
    dynamic_embed = model(data)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    target_embed = torch.randn(dynamic_embed.size())
    loss = loss_fn(dynamic_embed, target_embed)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    num_nodes = dynamic_embed.shape[0]
    embedding_dim = dynamic_embed.shape[1]

    print("Number of nodes:", num_nodes)
    print("Embedding dimension:", embedding_dim)

