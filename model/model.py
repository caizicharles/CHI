import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from .modules import *


class PredictionHead(nn.Module):

    def __init__(self,
                 attn_hidden_dim,
                 mlp_input_dim,
                 mlp_hidden_dim,
                 mlp_output_dim,
                 head_num=1,
                 ln=False,
                 mlp_act_layer=nn.GELU):
        super().__init__()

        self.Q = nn.Parameter(torch.Tensor(1, 1, mlp_input_dim))

        self.attn = MAB(attn_hidden_dim, attn_hidden_dim, mlp_input_dim, head_num, ln=ln)
        self.mlp = nn.Linear(mlp_input_dim, mlp_output_dim)
        # self.mlp = MLP(mlp_input_dim, mlp_hidden_dim, mlp_output_dim, mlp_act_layer)

    def forward(self, X):
        X = self.attn(self.Q.repeat(X.size(0), 1, 1), X)
        X = X.squeeze(-2)  # (B,D)
        X = self.mlp(X)  # (B,1)
        return X


class TimeTransformer(nn.Module):

    def __init__(self, hidden_dim, num_heads=4, depth=2, ln=False):
        super().__init__()

        self.attn = nn.ModuleList([SAB(2 * hidden_dim, 2 * hidden_dim, num_heads, ln=ln) for _ in range(depth)])

    def forward(self, X, visit_x, embed, attn_mask=None):
        embed = embed.repeat(1, 1, 2)  # (B,V,2D)
        X = torch.cat((X, visit_x), dim=-1)  # (B,V,2D)

        for _, block in enumerate(self.attn):
            X = block(X + embed, attn_mask)
        return X


class PrototypeLearner(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, depth=2, ln=False):
        super().__init__()

        # self.attn = nn.ModuleList([MAB(input_dim, input_dim, hidden_dim, num_heads, ln=ln)
        #                            for _ in range(depth)])
        self.attn = MAB(input_dim, input_dim, hidden_dim, num_heads, ln=ln)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, P):
        X = self.attn(X, P)
        # for _, block in enumerate(self.attn):
        #     P = block(X, P)
        X = self.fc(X)
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
        self.fc = nn.Linear(dim_hidden, dim_output)

    def forward(self, X, attn_mask=None):
        for _, block in enumerate(self.enc):
            X = block(X, attn_mask)
        for _, block in enumerate(self.dec):
            X = block(X, attn_mask)
        return self.fc(X)


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
                 time_depth=2,
                 node_embed=None,
                 edge_embed=None,
                 freeze=False):

        super().__init__()

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.gnn_layer = gnn_layer
        self.gnn_hidden_dim = gnn_hidden_dim
        self.full_node_ids = torch.arange(self.num_nodes + 1).to(device)
        self.gnn_hidden_dim = gnn_hidden_dim

        self.prototypes = nn.Parameter(torch.randn(proto_num, set_trans_out_dim))

        if node_embed is None:
            self.node_embed = nn.Embedding(num_nodes + 1, start_embed_dim, padding_idx=0)
        '''
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=freeze)

        if edge_embed is None:
            self.edge_embed = nn.Embedding(num_edges+1, embed_dim)
        else:
            self.edge_embed = nn.Embedding.from_pretrained(edge_embed, freeze=freeze)
        '''

        self.time_trans_embed = DataEmbedding(num_visits, max_weeks_between, proto_out_dim)

        self.fc = nn.Linear(start_embed_dim, gnn_hidden_dim)

        self.gnn = nn.ModuleList([GATConv(gnn_hidden_dim, gnn_hidden_dim) for _ in range(gnn_layer)])

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

        self.time_transformer = TimeTransformer(hidden_dim=time_trans_hidden_dim,
                                                num_heads=time_head_num,
                                                depth=time_depth)

        self.head = PredictionHead(attn_hidden_dim=2 * time_trans_hidden_dim,
                                   mlp_input_dim=2 * time_trans_hidden_dim,
                                   mlp_hidden_dim=2 * time_trans_hidden_dim,
                                   mlp_output_dim=out_dim)

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, visit_order, attn_mask):

        # print(f'node id shape: {node_ids.shape}')
        # print(f'edge idx shape: {edge_idx.shape}')
        # print(f'edge attr shape: {edge_attr.shape}')

        # full_edge_ids = torch.arange(self.num_edges)
        # edge_embed = self.edge_embed(full_edge_ids)
        # edge_embed = self.lin(edge_embed)

        x = self.node_embed(self.full_node_ids)  # (node_num,D)
        # x = self.fc(x)

        # Global GNN
        for layer in self.gnn:
            x = layer(x, edge_idx, edge_attr=edge_attr)

        B, V, N = node_ids.shape
        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)
        # TODO double check
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.gnn_hidden_dim))

        # Set Transformer
        x = self.set_transformer(x, attn_mask=attn_mask)  # (B,V,1,D)
        visit_x = x

        # Learn Prototypes
        prototypes = self.prototypes.unsqueeze(0).unsqueeze(0)
        prototypes = prototypes.repeat(B, V, 1, 1)  # (B,V,proto_num,D)
        x = self.prototype_learner(x, prototypes)  # (B,V,1,D)

        # Pos and Time Embedding
        embed = self.time_trans_embed(visit_order, visit_times)  # (B,V,D)

        # Time-Aware Transformer
        x = x.squeeze(-2)
        visit_x = visit_x.squeeze(-2)
        x = self.time_transformer(x, visit_x, embed, attn_mask=attn_mask)

        # Prediction
        out = self.head(x)

        return out, self.prototypes
