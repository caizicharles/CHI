import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from .modules import *


class SetTransformer(nn.Module):

    def __init__(self,
                 dim_input,
                 num_outputs,
                 dim_output,
                 num_inds=32,
                 dim_hidden=128,
                 num_heads=4,
                 encoder_depth=2,
                 decoder_depth=2,
                 ln=False):
        super(SetTransformer, self).__init__()

        self.enc = nn.ModuleList(
            [ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln) for _ in range(encoder_depth)])

        self.dec = nn.ModuleList([
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
        ])
        self.dec.extend([SAB(dim_hidden, dim_hidden, num_heads, ln=ln) for _ in range(decoder_depth)])
        self.dec.append(nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        for _, block in enumerate(self.enc):
            X = block(X)
        return self.dec(self.enc(X))


class OurModel(nn.Module):

    def __init__(self,
                 num_nodes,
                 num_edges,
                 embed_dim,
                 gnn_hidden_dim,
                 set_trans_hidden_dim,
                 set_trans_out_dim,
                 time_trans_hidden_dim,
                 time_trans_out_dim,
                 out_dim,
                 prototype_num=5,
                 set_num_heads=4,
                 set_encoder_depth=2,
                 set_decoder_depth=2,
                 set_trans_out_num=1,
                 time_num_heads=4,
                 time_encoder_depth=2,
                 time_decoder_depth=2,
                 gnn_layer=1,
                 node_embed=None,
                 edge_embed=None,
                 freeze=False):

        super().__init__()

        self.num_nodes = num_nodes
        # self.num_edges = num_edges
        self.gnn_layer = gnn_layer

        self.prototypes = nn.Parameter(torch.randn(1, prototype_num, set_trans_out_dim))

        if node_embed is None:
            self.node_embed = nn.Embedding(num_nodes+1, embed_dim, padding_idx=0)
        '''
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=freeze)

        if edge_embed is None:
            self.edge_embed = nn.Embedding(num_edges+1, embed_dim)
        else:
            self.edge_embed = nn.Embedding.from_pretrained(edge_embed, freeze=freeze)
        '''

        self.lin = nn.Linear(embed_dim, gnn_hidden_dim)
        self.head = nn.Linear(time_trans_out_dim, out_dim)

        self.gnn = nn.ModuleDict()
        for layer in range(1, gnn_layer + 1):
            self.gnn[str(layer)] = GATConv(gnn_hidden_dim, gnn_hidden_dim)
        
        self.set_transformer = SetTransformer(dim_input=gnn_hidden_dim,
                                              num_outputs=set_trans_out_num,
                                              dim_output=set_trans_out_dim,
                                              num_inds=1,
                                              dim_hidden=set_trans_hidden_dim,
                                              num_heads=set_num_heads,
                                              encoder_depth=time_encoder_depth,
                                              decoder_depth=set_decoder_depth)

    def forward(self, node_ids, edge_idx, edge_attr):

        # full_edge_ids = torch.arange(self.num_edges)
        # edge_embed = self.edge_embed(full_edge_ids)
        # edge_embed = self.lin(edge_embed)

        full_node_ids = torch.arange(self.num_nodes)
        x = self.node_embed(full_node_ids)
        x = self.lin(x)
        
        # Global GNN
        for layer in range(1, self.gnn_layer+1):
            x = self.gnn[str(layer)](x, edge_idx, edge_attr)

        x = x[:, node_ids]

        # Set Transformer
        x = self.set_transformer(x) # (B,1,D)

        # Learn Prototypes
        self.prototypes = self.prototypes.repeat(node_ids.size(0), 1, 1)    # (B,proto_num,D)
        x = torch.cat((x, self.prototypes), dim=1)
