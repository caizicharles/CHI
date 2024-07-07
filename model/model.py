import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from .modules import *


class TimeTransformet(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_heads=4,
                 depth=2,
                 ln=False):
        super().__init__()

        self.attn = None

    def forward(self, X, embed):

        return X


class PrototypeLearner(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_heads=4,
                 depth=2,
                 ln=False):
        super().__init__()

        # self.attn = nn.ModuleList([MAB(input_dim, input_dim, hidden_dim, num_heads, ln=ln)
        #                            for _ in range(depth)])
        self.attn = MAB(input_dim, input_dim, hidden_dim, num_heads, ln=ln)
        self.ffn = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, P):
        X = self.attn(X, P)
        # for _, block in enumerate(self.attn):
        #     P = block(X, P)
        X = self.ffn(X)
        return X


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
        super().__init__()

        self.enc = nn.ModuleList(
            [ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln) for _ in range(encoder_depth)])

        self.dec = nn.ModuleList([
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
        ])
        self.dec.extend([SAB(dim_hidden, dim_hidden, num_heads, ln=ln) for _ in range(decoder_depth)])
        self.proj = nn.Linear(dim_hidden, dim_output)

    def forward(self, X, attn_mask=None):
        for _, block in enumerate(self.enc):
            X = block(X, attn_mask)
        for _, block in enumerate(self.dec):
            X = block(X, attn_mask)
        return self.proj(X)


class OurModel(nn.Module):

    def __init__(self,
                 device,
                 num_nodes,
                 num_edges,
                 num_visits,
                 max_weeks_between,
                 start_embed_dim,
                 gnn_hidden_dim,
                 set_trans_hidden_dim,
                 set_trans_out_dim,
                 proto_hidden_dim,
                 proto_out_dim,
                 time_trans_hidden_dim,
                 time_trans_out_dim,
                 out_dim,
                 gnn_layer=1,
                 proto_num=5,
                 set_head_num=4,
                 set_num_inds=32,
                 set_encoder_depth=2,
                 set_decoder_depth=2,
                 set_trans_out_num=1,
                 proto_head_num=4,
                 proto_depth=2,
                 time_head_num=4,
                 time_encoder_depth=2,
                 time_decoder_depth=2,
                 node_embed=None,
                 edge_embed=None,
                 freeze=False):

        super().__init__()

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.gnn_layer = gnn_layer
        self.gnn_hidden_dim = gnn_hidden_dim
        self.full_node_ids = torch.arange(self.num_nodes+1).to(device)
        self.gnn_hidden_dim = gnn_hidden_dim

        self.prototypes = nn.Parameter(torch.randn(1, 1, proto_num, set_trans_out_dim))

        if node_embed is None:
            self.node_embed = nn.Embedding(num_nodes+1, start_embed_dim, padding_idx=0)
        '''
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=freeze)

        if edge_embed is None:
            self.edge_embed = nn.Embedding(num_edges+1, embed_dim)
        else:
            self.edge_embed = nn.Embedding.from_pretrained(edge_embed, freeze=freeze)
        '''

        self.time_trans_embed = DataEmbedding(num_visits, max_weeks_between, proto_out_dim)

        self.lin = nn.Linear(start_embed_dim, gnn_hidden_dim)
        self.head = nn.Linear(time_trans_out_dim, out_dim)

        self.gnn = nn.ModuleDict()
        for layer in range(1, gnn_layer + 1):
            self.gnn[str(layer)] = GATConv(gnn_hidden_dim, gnn_hidden_dim)

        self.set_transformer = SetTransformer(dim_input=gnn_hidden_dim,
                                              num_outputs=set_trans_out_num,
                                              dim_output=set_trans_out_dim,
                                              num_inds=set_num_inds,
                                              dim_hidden=set_trans_hidden_dim,
                                              num_heads=set_head_num,
                                              encoder_depth=set_encoder_depth,
                                              decoder_depth=set_decoder_depth)

        self.prototype_learner = PrototypeLearner(input_dim=set_trans_out_dim,
                                                  hidden_dim=proto_hidden_dim,
                                                  output_dim=proto_out_dim,
                                                  num_heads=proto_head_num,
                                                  depth=proto_depth)

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, visit_order):

        print(f'node id shape: {node_ids.shape}')
        print(f'edge idx shape: {edge_idx.shape}')
        print(f'edge attr shape: {edge_attr.shape}')

        # full_edge_ids = torch.arange(self.num_edges)
        # edge_embed = self.edge_embed(full_edge_ids)
        # edge_embed = self.lin(edge_embed)        

        x = self.node_embed(self.full_node_ids)     # (node_num,D)
        x = self.lin(x)

        # Global GNN
        for layer in range(1, self.gnn_layer + 1):
            x = self.gnn[str(layer)](x, edge_idx, edge_attr=edge_attr)

        B, V, N = node_ids.shape
        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.gnn_hidden_dim))

        zero_elements = node_ids == 0
        zero_rows = zero_elements.all(dim=-1)

        # Set Transformer
        x = self.set_transformer(x, attn_mask=zero_rows)  # (B,V,1,D)

        # Learn Prototypes
        prototypes = self.prototypes.repeat(B, V, 1, 1)  # (B,V,proto_num,D)
        x = self.prototype_learner(x, prototypes)

        # Pos and Time Embedding
        embed = self.time_trans_embed(visit_order, visit_times)
        embed = embed.unsqueeze(-2)

        # Time-Aware Transformer
        