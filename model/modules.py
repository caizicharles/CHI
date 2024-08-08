import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch import Tensor
from typing import Callable, Optional, Union
import math

########## AdaCare ##########


class DeeprLayer(nn.Module):

    def __init__(self, feature_size: int = 100, window: int = 1, hidden_size: int = 3):
        super().__init__()

        self.hidden_size = hidden_size
        self.conv = torch.nn.Conv1d(feature_size, hidden_size, kernel_size=2 * window + 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        if mask is not None:
            x = x * mask.unsqueeze(-1).unsqueeze(-1)

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


class BiAttentionGNNConv(MessagePassing):

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


class DataEmbedding(nn.Module):

    def __init__(self, pos_embed_num, time_embed_num, embed_dim, dropout=0.1):
        super().__init__()

        self.pos_embed = PositionalEmbedding(pos_embed_num, embed_dim)
        self.time_embed = TemporalEmbedding(time_embed_num, embed_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, visit_order, visit_rel_times):
        embed = self.pos_embed(visit_order) + self.time_embed(visit_rel_times)

        return embed  # self.dropout(embed)


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
        O = torch.cat((Q_ + A.matmul(V_)).split(Q.size(0), 0), dim=-1)
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
        nn.init.xavier_uniform_(self.Q)

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
        # TODO only + embed when input
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
