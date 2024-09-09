import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import MessagePassing as MP
from torch_geometric.utils import degree
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min
from torch_geometric.typing import (Adj, OptPairTensor, OptTensor, Size)
from torch import Tensor
from typing import Callable, Optional, Union
import math

########## Deepr ##########


class TimeGapEmbedding(nn.Module):

    def __init__(self, embed_dim, device):
        super().__init__()

        self.boundary = torch.tensor([1, 3, 6, 12], dtype=torch.float32, device=device)
        self.time_embed = nn.Embedding(5, embed_dim)

    def forward(self, visit_rel_times):

        visit_rel_times = visit_rel_times / 4
        indices = torch.bucketize(visit_rel_times, self.boundary, right=True)

        return self.time_embed(indices)


class DeeprLayer(nn.Module):

    def __init__(self, feature_size: int = 100, window: int = 1, hidden_size: int = 3):
        super().__init__()

        self.hidden_size = hidden_size
        self.conv = torch.nn.Conv1d(feature_size, hidden_size, kernel_size=2 * window + 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        if mask is not None:
            x = x * mask

        B, V, N, D = x.shape
        x = x.reshape(B * V, N, D)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = x.max(dim=-1)[0]
        x = x.reshape(B, V, self.hidden_size)
        x = x.max(dim=-2)[0]

        return x


########## AdaCare ##########


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        super().__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input, device='cuda'):
        original_size = input.size()
        input = input.view(-1, input.size(self.dim))

        dim = 1
        number_of_logits = input.size(dim)

        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, device=device, dtype=torch.float32).view(1, -1)
        range = range.expand_as(zs)

        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        zs_sparse = is_gt * zs
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        self.output = torch.max(torch.zeros_like(input), input - taus)

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


class CausalConv1d(torch.nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super().__init__(in_channels,
                         out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=self.__padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class Recalibration(nn.Module):

    def __init__(self, channel, reduction=9, use_h=True, use_c=True, activation='sigmoid'):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.use_h = use_h
        self.use_c = use_c
        scale_dim = 0
        self.activation = activation

        self.nn_c = nn.Linear(channel, channel // reduction)
        scale_dim += channel // reduction

        self.nn_rescale = nn.Linear(scale_dim, channel)
        self.sparsemax = Sparsemax(dim=1)

    def forward(self, x, device='cuda'):
        b, c, t = x.size()

        y_origin = x[:, :, -1]
        se_c = self.nn_c(y_origin)
        se_c = torch.relu(se_c)
        y = se_c

        y = self.nn_rescale(y).view(b, c, 1)
        if self.activation == 'sigmoid':
            y = torch.sigmoid(y)
        else:
            y = self.sparsemax(y, device)
        return x * y.expand_as(x), y


########## GraphCare ##########


class BiAttentionGNNConv(MP):

    def __init__(self,
                 nn: torch.nn.Module,
                 eps: float = 0.,
                 train_eps: bool = False,
                 edge_dim: Optional[int] = None,
                 edge_attn=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.nn = nn
        self.initial_eps = eps
        self.edge_attn = edge_attn
        if edge_attn:
            # self.W_R = torch.nn.Linear(edge_dim, edge_dim)
            self.W_R = torch.nn.Linear(edge_dim, 1)
        else:
            self.W_R = None

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        self.nn.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        if self.W_R is not None:
            self.W_R.reset_parameters()

    def forward(self,
                x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                edge_attr: OptTensor = None,
                size: Size = None,
                attn: Tensor = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size, attn=attn)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        if self.W_R is not None:
            w_rel = self.W_R(edge_attr)
        else:
            w_rel = None

        return self.nn(out), w_rel

    def message(self, x_j: Tensor, edge_attr: Tensor, attn: Tensor) -> Tensor:

        if self.edge_attn:
            w_rel = self.W_R(edge_attr)
            out = (x_j * attn + w_rel * edge_attr).relu()
        else:
            out = (x_j * attn).relu()
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


def masked_softmax(src: Tensor, mask: Tensor, dim: int = -1) -> Tensor:

    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)

    return out + 1e-8  # Add small constant to avoid numerical issues


########## OurModal ##########


class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = None,
                 output_dim: int = None,
                 act_layer=nn.GELU,
                 dropout_prob: float = 0.):
        super().__init__()

        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PositionalEmbedding(nn.Module):

    def __init__(self, embed_num, embed_dim):
        super().__init__()

        self.pos_embed = nn.Embedding(embed_num, embed_dim)

    def forward(self, visit_order):

        return self.pos_embed(visit_order)


class TemporalEmbedding(nn.Module):

    def __init__(self, embed_num, embed_dim):
        super().__init__()

        self.time_embed = nn.Embedding(embed_num, embed_dim, padding_idx=0)

    def forward(self, visit_rel_times):

        return self.time_embed(visit_rel_times)


class CodeTypeEmbedding(nn.Module):

    def __init__(self, embed_num, embed_dim, padding_idx=0):
        super().__init__()

        self.type_embed = nn.Embedding(embed_num, embed_dim, padding_idx=padding_idx)

    def forward(self, visit_node_type):

        return self.type_embed(visit_node_type)


class DataEmbedding(nn.Module):

    def __init__(self, pos_embed_num, time_embed_num, embed_dim, dropout=0.1):
        super().__init__()

        self.pos_embed = PositionalEmbedding(pos_embed_num, embed_dim)
        self.time_embed = TemporalEmbedding(time_embed_num, embed_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, visit_order, visit_rel_times):
        # embed = self.pos_embed(visit_order) + self.time_embed(visit_rel_times)
        embed = self.time_embed(visit_rel_times)
        return embed


class CompGCNConv(MessagePassing):

    eps = 1e-6

    def __init__(self,
                 input_dim,
                 output_dim,
                 message_func="rotate",
                 aggregate_func="pna",
                 activation="relu",
                 layer_norm=False,
                 use_rel_update=True,
                 use_dir_weight=True,
                 use_norm=True,
                 num_relations=None):
        super(CompGCNConv, self).__init__(flow="target_to_source",
                                          aggr=aggregate_func if aggregate_func != "pna" else None)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.use_norm = use_norm
        self.num_relations = num_relations

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 12, output_dim)
        if self.message_func == "mlp":
            self.msg_mlp = nn.Sequential(nn.Linear(2 * input_dim, input_dim), nn.ReLU(),
                                         nn.Linear(input_dim, input_dim))

        self.use_rel_update = use_rel_update
        self.use_dir_weight = use_dir_weight

        if self.use_rel_update:
            self.relation_update = nn.Linear(input_dim, input_dim)

        # CompGCN weight matrices
        if self.use_dir_weight:
            self.w_in = nn.Parameter(torch.empty(input_dim, output_dim))
            self.w_out = nn.Parameter(torch.empty(input_dim, output_dim))
            self.w_loop = nn.Parameter(torch.empty(input_dim, output_dim))
            nn.init.xavier_uniform_(self.w_in)
            nn.init.xavier_uniform_(self.w_out)
            nn.init.xavier_uniform_(self.w_loop)
        else:
            self.w = nn.Parameter(torch.empty(input_dim, output_dim))
            nn.init.xavier_uniform_(self.w)

        # layer-specific self-loop relation parameter
        self.loop_relation = nn.Parameter(torch.empty(1, input_dim))
        nn.init.xavier_uniform_(self.loop_relation)

    def forward(self, x, edge_index, edge_type, relation_embs):
        """
        CompGCN forward pass is the average of direct, inverse, and self-loop messages
        """

        # out graph -> the original graph without inverse edges
        # edge_list = edge_index

        # in PyG Entities datasets, direct edges have even indices, inverse - odd
        if self.use_dir_weight:
            # out_index = edge_list[:, edge_type % 2 == 0]
            # out_type = edge_type[edge_type % 2 == 0]
            # out_norm = self.compute_norm(out_index, x.shape[0]) if self.use_norm else torch.ones_like(out_type, dtype=torch.float)

            # in graph -> the graph with only inverse edges
            # in_index = edge_list[:, edge_type % 2 == 1]
            # in_type = edge_type[edge_type % 2 == 1]
            # in_norm = self.compute_norm(in_index, x.shape[0]) if self.use_norm else torch.ones_like(in_type, dtype=torch.float)
            in_norm = self.compute_norm(edge_index, x.shape[0]) if self.use_norm else torch.ones_like(edge_type,
                                                                                                      dtype=torch.float)

            # self_loop graph -> the graph with only self-loop relation type
            # node_in = node_out = torch.arange(x.shape[0], device=x.device)
            # loop_index = torch.stack([node_in, node_out], dim=0)
            # loop_type = torch.zeros(loop_index.shape[1], dtype=torch.long, device=x.device)

            # message passing
            # out_update = self.propagate(x=x, edge_index=out_index, edge_type=out_type, relation_embs=relation_embs, relation_weight=self.w_out, edge_weight=out_norm)
            output = self.propagate(edge_index,
                                    x=x,
                                    edge_type=edge_type,
                                    relation_embs=relation_embs,
                                    relation_weight=self.w_in,
                                    edge_weight=in_norm)
            # loop_update = self.propagate(x=x, edge_index=loop_index, edge_type=loop_type, relation_embs=self.loop_relation, relation_weight=self.w_loop)

            # output = (out_update + in_update + loop_update) / 3.0

        else:
            # add self-loops
            node_in = node_out = torch.arange(x.shape[0], device=x.device)
            loop_index = torch.stack([node_in, node_out], dim=0)
            edge_index = torch.cat([edge_index, loop_index], dim=-1)

            loop_type = torch.zeros(loop_index.shape[1], dtype=torch.long, device=x.device).fill_(len(relation_embs))
            edge_type = torch.cat([edge_type, loop_type], dim=-1)
            relation_embs = torch.cat([relation_embs, self.loop_relation], dim=0)

            norm = self.compute_norm(edge_index, num_ent=x.shape[0]) if self.use_norm else torch.ones_like(
                edge_type, dtype=torch.float)
            output = self.propagate(x=x,
                                    edge_index=edge_index,
                                    edge_type=edge_type,
                                    relation_embs=relation_embs,
                                    relation_weight=self.w,
                                    edge_weight=norm)

        if self.use_rel_update:
            relation_embs = self.relation_update(relation_embs)

        return output, relation_embs

    def message(self, x_j, edge_type, relation_embs, relation_weight, edge_weight=None):

        edge_input = relation_embs[edge_type]

        if self.message_func == "transe":
            message = edge_input + x_j
        elif self.message_func == "distmult":
            message = edge_input * x_j
        elif self.message_func == "rotate":
            node_re, node_im = x_j.chunk(2, dim=-1)
            edge_re, edge_im = edge_input.chunk(2, dim=-1)
            message_re = node_re * edge_re - node_im * edge_im
            message_im = node_re * edge_im + node_im * edge_re
            message = torch.cat([message_re, message_im], dim=-1)
        elif self.message_func == "mlp":
            message = self.msg_mlp(torch.cat([x_j, edge_input], dim=-1))
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        # message transformation: can be direction-wise or simple linear map
        message = torch.mm(message, relation_weight)

        return message if edge_weight is None else message * edge_weight.view(-1, 1)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):

        if self.aggregate_func != "pna":
            return super().aggregate(inputs=inputs, index=index, ptr=ptr, dim_size=dim_size)
        else:
            mean = scatter_mean(inputs, index, dim=0, dim_size=dim_size)
            sq_mean = scatter_mean(inputs**2, index, dim=0, dim_size=dim_size)
            max = scatter_max(inputs, index, dim=0, dim_size=dim_size)[0]
            min = scatter_min(inputs, index, dim=0, dim_size=dim_size)[0]
            std = (sq_mean - mean**2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)

            deg = degree(index, dim_size, dtype=inputs.dtype)
            scale = (deg + 1).log()
            scale = scale / scale.mean()
            scales = torch.stack([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)

        return update

    def update(self, inputs):
        # in CompGCN, we just return updated states, no aggregation with inputs
        # update = update
        output = inputs if self.aggregate_func != "pna" else self.linear(inputs)
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    @staticmethod
    def compute_norm(edge_index, num_ent):
        """
        Re-normalization trick used by GCN-based architectures without attention.
        """
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()  # Identity matrix where we know all entities are there
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # Summing number of weights of
        # the edges, D = A + I
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0  # for numerical stability
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # Norm parameter D^{-0.5} *

        return norm


class MAB(nn.Module):

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, skip=True, ln=False):
        super().__init__()

        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.skip = skip
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, attn_mask=None, return_scores=False):

        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        if attn_mask is not None:
            # attn_mask: (B,V,N)
            if K.dim() == 3:
                attn_mask = attn_mask.all(dim=-1)  # Visit Mask (B,V)

            if Q.size(-2) == attn_mask.size(-1):
                Q.masked_fill_(attn_mask.unsqueeze(-1), value=0)
            if K.size(-2) == attn_mask.size(-1):
                K.masked_fill_(attn_mask.unsqueeze(-1), value=0)
                V.masked_fill_(attn_mask.unsqueeze(-1), value=0)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, -1), dim=0)
        K_ = torch.cat(K.split(dim_split, -1), dim=0)
        V_ = torch.cat(V.split(dim_split, -1), dim=0)

        scores = Q_.matmul(K_.transpose(-2, -1)) / math.sqrt(dim_split)
        A = torch.softmax(scores, dim=-1)

        if self.skip:
            O = torch.cat((Q_ + A.matmul(V_)).split(Q.size(0), 0), dim=-1)
        else:
            O = torch.cat(A.matmul(V_).split(Q.size(0), 0), dim=-1)

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        if return_scores:
            return O, A
        else:
            return O, None


class SAB(nn.Module):

    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()

        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, x, attn_mask=None, return_scores=False):
        return self.mab(x, x, attn_mask, return_scores=return_scores)


class ISAB(nn.Module):

    def __init__(self, dim_in, dim_out, num_heads, num_visits, num_inds, qkv_length=4, ln=False):
        super().__init__()

        self.qkv_length = qkv_length

        if self.qkv_length == 4:
            self.I = nn.Parameter(torch.Tensor(1, num_visits, num_inds, dim_out))
        elif self.qkv_length == 3:
            self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)

        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, attn_mask=None, return_scores=False):

        if self.qkv_length == 4:
            Q = self.I.repeat(X.size(0), 1, 1, 1)
        elif self.qkv_length == 3:
            Q = self.I.repeat(X.size(0), 1, 1)

        H, _ = self.mab0(Q, X, attn_mask)
        return self.mab1(X, H, attn_mask, return_scores=return_scores)


class PMA(nn.Module):

    def __init__(self, dim, num_heads, num_visits, num_seeds, ln=False):
        super().__init__()

        self.S = nn.Parameter(torch.Tensor(1, num_visits, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)

        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, attn_mask=None, return_scores=False):

        Q = self.S.repeat(X.size(0), 1, 1, 1)
        return self.mab(Q, X, attn_mask, return_scores=return_scores)


class ImportanceAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads=2, depth=1, ln=False):
        super().__init__()

        self.blocks = nn.ModuleList([MAB(hidden_dim, hidden_dim, hidden_dim, num_heads, ln=ln) for _ in range(depth)])

    def forward(self, x, p, return_scores=False):

        p = F.normalize(p, p=2, dim=-1)

        for _, block in enumerate(self.blocks):
            x, scores = block(x, p, return_scores=return_scores)

        return x, scores


class PrototypeLearner(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, skip=False, ln=False):
        super().__init__()

        self.attn = MAB(input_dim, input_dim, hidden_dim, num_heads, skip=skip, ln=ln)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, p, return_scores=False):

        p = p.unsqueeze(0)
        p = p.repeat(x.size(0), 1, 1)
        p = F.normalize(p, p=2, dim=-1)

        x, scores = self.attn(x, p, return_scores=return_scores)
        x = self.fc(x)

        return x, scores


class SetTransformer(nn.Module):

    def __init__(self,
                 dim_input,
                 num_outputs,
                 dim_output,
                 num_visits,
                 num_inds=32,
                 dim_hidden=128,
                 num_heads=4,
                 encoder_depth=2,
                 decoder_depth=2,
                 qkv_length=4,
                 ln=False):
        super().__init__()

        self.enc = nn.ModuleList([
            ISAB(dim_input, dim_hidden, num_heads, num_visits, num_inds, qkv_length, ln=ln)
            for _ in range(encoder_depth)
        ])

        self.dec = nn.ModuleList([
            PMA(dim_hidden, num_heads, num_visits, num_outputs, ln=ln),
        ])
        self.dec.extend([SAB(dim_hidden, dim_hidden, num_heads, ln=ln) for _ in range(decoder_depth)])
        self.fc = nn.Linear(dim_hidden, dim_output)

    def forward(self, x, embed=None, attn_mask=None, return_scores=False):

        if embed is not None:
            x += embed
        for _, block in enumerate(self.enc):
            x, enc_scores = block(x, attn_mask=attn_mask, return_scores=return_scores)
        for _, block in enumerate(self.dec):
            x, dec_scores = block(x, attn_mask=attn_mask, return_scores=return_scores)

        return self.fc(x), enc_scores, dec_scores
