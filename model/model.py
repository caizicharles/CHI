import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from .modules import *


class OurModel(nn.Module):

    def __init__(self,
                 num_nodes,
                 num_edges,
                 embed_dim,
                 gnn_hidden_dim,
                 trans_hidden_dim,
                 out_dim,
                 node_embed=None,
                 edge_embed=None,
                 gnn_layer=1,
                 freeze=False):

        super().__init__()

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.embed_dim = embed_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_layer = gnn_layer

        if node_embed is None:
            self.node_embed = nn.Embedding(num_nodes, embed_dim)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=freeze)

        if edge_embed is None:
            self.edge_embed = nn.Embedding(num_edges, embed_dim)
        else:
            self.edge_embed = nn.Embedding.from_pretrained(edge_embed, freeze=freeze)

        self.lin = nn.Linear(embed_dim, gnn_hidden_dim)
        self.head = nn.Linear(trans_hidden_dim, out_dim)
        self.gnn = nn.ModuleDict()

        for layer in range(1, gnn_layer + 1):
            self.gnn[str(layer)] = GATConv(gnn_hidden_dim, gnn_hidden_dim)

    def forward(self, node_ids, edge_idx):

        full_node_ids = torch.arange(self.num_nodes)
        full_edge_ids = torch.arange(self.num_edges)
        x = self.node_embed(full_node_ids)
        edge_embed = self.edge_embed(full_edge_ids)

        x = self.lin(x)
        edge_embed = self.lin(edge_embed)

        # for layer in range(1, self.gnn_layer+1):
        #     x = self.gnn[str(layer)](x, edge_idx, edge_embed)
