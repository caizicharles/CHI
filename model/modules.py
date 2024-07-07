import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, embed_num, embed_dim):
        super().__init__()

        self.pos_embed = nn.Embedding(embed_num, embed_dim)

    def forward(self, visit_order):
        
        return self.pos_embed(visit_order)


class TemporalEmbedding(nn.Module):

    def __init__(self,
                 embed_num,
                 embed_dim):
        super().__init__()

        self.time_embed = nn.Embedding(embed_num, embed_dim, padding_idx=0)

    def forward(self, visit_rel_times):
        
        return self.time_embed(visit_rel_times)


class DataEmbedding(nn.Module):
    def __init__(self, pos_embed_num, time_embed_num, embed_dim, dropout=0.1):
        super().__init__()

        self.pos_embed = PositionalEmbedding(pos_embed_num, embed_dim)
        self.time_embed = TemporalEmbedding(time_embed_num, embed_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, visit_order, visit_rel_times):
        embed = self.pos_embed(visit_order) + self.time_embed(visit_rel_times)
        
        return embed # self.dropout(embed)


class MAB(nn.Module):

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()

        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, attn_mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        last_dim = len(Q.shape)-1

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, last_dim), 0)
        K_ = torch.cat(K.split(dim_split, last_dim), 0)
        V_ = torch.cat(V.split(dim_split, last_dim), 0)

        _, _, nq, _ = Q_.shape
        bh, v, nk, _ = K_.shape

        scores = Q_.matmul(K_.transpose(2, 3)) / math.sqrt(dim_split)

        if attn_mask is not None:
            # attn_mask (B, V) -> (BH, V, NQ, NK)
            attn_mask = attn_mask.repeat(self.num_heads, 1)
            attn_mask = attn_mask.unsqueeze(-1).unsqueeze(-1)
            attn_mask = attn_mask.expand(bh, v, nq, nk)
            scores = scores.masked_fill(attn_mask == True, float('-inf'))

        A = torch.softmax(scores, last_dim)
        O = torch.cat((Q_ + A.matmul(V_)).split(Q.size(0), 0), last_dim)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):

    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()

        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, attn_mask=None):
        return self.mab(X, X, attn_mask)


class ISAB(nn.Module):

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()

        self.I = nn.Parameter(torch.Tensor(1, num_inds, 1, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, attn_mask=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1, 1), X, attn_mask)
        return self.mab1(X, H, attn_mask)


class PMA(nn.Module):

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()

        self.S = nn.Parameter(torch.Tensor(1, num_seeds, 1, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, attn_mask=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1, 1), X, attn_mask)
