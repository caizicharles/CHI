import random
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import MessagePassing as MP
from torch_geometric.utils import degree
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min
from torch_geometric.typing import (Adj, OptPairTensor, OptTensor, Size)
from torch import Tensor
from typing import Callable, Optional, Union, Tuple
from pyhealth.models import RNNLayer
from pyhealth.models.utils import get_last_visit
from sklearn.neighbors import kneighbors_graph



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
    

########## GRASP ##########


class FinalAttentionQKV(nn.Module):
    def __init__(
        self,
        attention_input_dim: int,
        attention_hidden_dim: int,
        attention_type: str = "add",
        dropout: float = 0.5,
    ):
        super(FinalAttentionQKV, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim

        self.W_q = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_k = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_v = nn.Linear(attention_input_dim, attention_hidden_dim)

        self.W_out = nn.Linear(attention_hidden_dim, 1)

        self.b_in = nn.Parameter(
            torch.zeros(
                1,
            )
        )
        self.b_out = nn.Parameter(
            torch.zeros(
                1,
            )
        )

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out.weight, a=math.sqrt(5))

        self.Wh = nn.Parameter(
            torch.randn(2 * attention_input_dim, attention_hidden_dim)
        )
        self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
        self.ba = nn.Parameter(
            torch.zeros(
                1,
            )
        )

        nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        (
            batch_size,
            time_step,
            input_dim,
        ) = input.size()  # batch_size * input_dim + 1 * hidden_dim(i)
        input_q = self.W_q(input[:, -1, :])  # b h
        input_k = self.W_k(input)  # b t h
        input_v = self.W_v(input)  # b t h

        if self.attention_type == "add":  # B*T*I  @ H*I

            q = torch.reshape(
                input_q, (batch_size, 1, self.attention_hidden_dim)
            )  # B*1*H
            h = q + input_k + self.b_in  # b t h
            h = self.tanh(h)  # B*T*H
            e = self.W_out(h)  # b t 1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == "mul":
            q = torch.reshape(
                input_q, (batch_size, self.attention_hidden_dim, 1)
            )  # B*h 1
            e = torch.matmul(input_k, q).squeeze()  # b t

        elif self.attention_type == "concat":
            q = input_q.unsqueeze(1).repeat(1, time_step, 1)  # b t h
            k = input_k
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t
        else:
            raise ValueError(
                "Unknown attention type: {}, please use add, mul, concat".format(
                    self.attention_type
                )
            )

        a = self.softmax(e)  # B*T
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze()  # B*I

        return v, a


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x)))), None


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pos = self.pe[:, : x.size(1)].clone().requires_grad_(False)
        x = x + pos
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList(
            [nn.Linear(d_model, self.d_k * self.h) for _ in range(3)]
        )
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)  # b h t d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # b h t t
        if mask is not None:  # 1 1 t t
            scores = scores.masked_fill(mask == 0, -1e9)  # b h t t 下三角
        p_attn = torch.softmax(scores, dim=-1)  # b h t t
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn  # b h t v (d_k)

    def cov(self, m, y=None):
        if y is not None:
            m = torch.cat((m, y), dim=0)
        m_exp = torch.mean(m, dim=1)
        x = m - m_exp[:, None]
        cov = 1 / (x.size(1) - 1) * x.mm(x.t())
        return cov

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # 1 1 t t

        nbatches = query.size(0)  # b
        input_dim = query.size(1)  # i+1
        feature_dim = query.size(1)  # i+1

        # input size -> # batch_size * d_input * hidden_dim

        # d_model => h * d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]  # b num_head d_input d_k

        x, self.attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout
        )  # b num_head d_input d_v (d_k)

        x = (
            x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        )  # batch_size * d_input * hidden_dim

        # DeCov
        DeCov_contexts = x.transpose(0, 1).transpose(1, 2)  # I+1 H B
        Covs = self.cov(DeCov_contexts[0, :, :])
        DeCov_loss = 0.5 * (
            torch.norm(Covs, p="fro") ** 2 - torch.norm(torch.diag(Covs)) ** 2
        )
        for i in range(feature_dim - 1):
            Covs = self.cov(DeCov_contexts[i + 1, :, :])
            DeCov_loss += 0.5 * (
                torch.norm(Covs, p="fro") ** 2 - torch.norm(torch.diag(Covs)) ** 2
            )

        return self.final_linear(x), DeCov_loss


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-7):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        returned_value = sublayer(self.norm(x))
        return x + self.dropout(returned_value[0]), returned_value[1]


class SingleAttention(nn.Module):
    def __init__(
        self,
        attention_input_dim,
        attention_hidden_dim,
        attention_type="add",
        time_aware=False,
    ):
        super(SingleAttention, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim
        self.time_aware = time_aware

        # batch_time = torch.arange(0, batch_mask.size()[1], dtype=torch.float32).reshape(1, batch_mask.size()[1], 1)
        # batch_time = batch_time.repeat(batch_mask.size()[0], 1, 1)

        if attention_type == "add":
            if self.time_aware:
                # self.Wx = nn.Parameter(torch.randn(attention_input_dim+1, attention_hidden_dim))
                self.Wx = nn.Parameter(
                    torch.randn(attention_input_dim, attention_hidden_dim)
                )
                self.Wtime_aware = nn.Parameter(torch.randn(1, attention_hidden_dim))
                nn.init.kaiming_uniform_(self.Wtime_aware, a=math.sqrt(5))
            else:
                self.Wx = nn.Parameter(
                    torch.randn(attention_input_dim, attention_hidden_dim)
                )
            self.Wt = nn.Parameter(
                torch.randn(attention_input_dim, attention_hidden_dim)
            )
            self.bh = nn.Parameter(
                torch.zeros(
                    attention_hidden_dim,
                )
            )
            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wd, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == "mul":
            self.Wa = nn.Parameter(
                torch.randn(attention_input_dim, attention_input_dim)
            )
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == "concat":
            if self.time_aware:
                self.Wh = nn.Parameter(
                    torch.randn(2 * attention_input_dim + 1, attention_hidden_dim)
                )
            else:
                self.Wh = nn.Parameter(
                    torch.randn(2 * attention_input_dim, attention_hidden_dim)
                )

            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        elif attention_type == "new":
            self.Wt = nn.Parameter(
                torch.randn(attention_input_dim, attention_hidden_dim)
            )
            self.Wx = nn.Parameter(
                torch.randn(attention_input_dim, attention_hidden_dim)
            )

            self.rate = nn.Parameter(torch.zeros(1) + 0.8)
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))

        else:
            raise RuntimeError(
                "Wrong attention type. Please use 'add', 'mul', 'concat' or 'new'."
            )

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, input, mask, device):

        (
            batch_size,
            time_step,
            input_dim,
        ) = input.size()  # batch_size * time_step * hidden_dim(i)

        time_decays = (
            torch.tensor(range(time_step - 1, -1, -1), dtype=torch.float32)
            .unsqueeze(-1)
            .unsqueeze(0)
            .to(device=device)
        )  # 1*t*1
        b_time_decays = time_decays.repeat(batch_size, 1, 1) + 1  # b t 1

        if self.attention_type == "add":  # B*T*I  @ H*I
            last_visit = get_last_visit(input, mask)
            q = torch.matmul(last_visit, self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            if self.time_aware == True:
                k = torch.matmul(input, self.Wx)  # b t h
                time_hidden = torch.matmul(b_time_decays, self.Wtime_aware)  # b t h
            else:
                k = torch.matmul(input, self.Wx)  # b t h
            h = q + k + self.bh  # b t h
            if self.time_aware:
                h += time_hidden
            h = self.tanh(h)  # B*T*H
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t
        elif self.attention_type == "mul":
            last_visit = get_last_visit(input, mask)
            e = torch.matmul(last_visit, self.Wa)  # b i
            e = (
                torch.matmul(e.unsqueeze(1), input.permute(0, 2, 1)).reshape(
                    batch_size, time_step
                )
                + self.ba
            )  # b t
        elif self.attention_type == "concat":
            last_visit = get_last_visit(input, mask)
            q = last_visit.unsqueeze(1).repeat(1, time_step, 1)  # b t i
            k = input
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            if self.time_aware:
                c = torch.cat((c, b_time_decays), dim=-1)  # B*T*2I+1
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == "new":
            last_visit = get_last_visit(input, mask)
            q = torch.matmul(last_visit, self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            k = torch.matmul(input, self.Wx)  # b t h
            dot_product = torch.matmul(q, k.transpose(1, 2)).reshape(
                batch_size, time_step
            )  # b t
            denominator = self.sigmoid(self.rate) * (
                torch.log(2.72 + (1 - self.sigmoid(dot_product)))
                * (b_time_decays.reshape(batch_size, time_step))
            )
            e = self.relu(self.sigmoid(dot_product) / (denominator))  # b * t
        else:
            raise ValueError(
                "Wrong attention type. Plase use 'add', 'mul', 'concat' or 'new'."
            )

        if mask is not None:
            e = e.masked_fill(mask == 0, -1e9)
        a = self.softmax(e)  # B*T
        v = torch.matmul(a.unsqueeze(1), input).reshape(batch_size, input_dim)  # B*I

        return v, a
 

class ConCareLayer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        hidden_dim: int = 128,
        num_head: int = 4,
        pe_hidden: int = 64,
        dropout: int = 0.5,
    ):
        super(ConCareLayer, self).__init__()

        # hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # d_model
        self.transformer_hidden = hidden_dim
        self.num_head = num_head
        self.pe_hidden = pe_hidden
        # self.output_dim = output_dim
        self.dropout = dropout
        self.static_dim = static_dim

        # layers
        self.PositionalEncoding = PositionalEncoding(
            self.transformer_hidden, dropout=0, max_len=400
        )

        self.GRUs = nn.ModuleList(
            [
                nn.GRU(1, self.hidden_dim, batch_first=True)
                for _ in range(self.input_dim)
            ]
        )
        self.LastStepAttentions = nn.ModuleList(
            [
                SingleAttention(
                    self.hidden_dim,
                    8,
                    attention_type="new",
                    time_aware=True,
                )
                for _ in range(self.input_dim)
            ]
        )

        self.FinalAttentionQKV = FinalAttentionQKV(
            self.hidden_dim,
            self.hidden_dim,
            attention_type="mul",
            dropout=self.dropout,
        )

        self.MultiHeadedAttention = MultiHeadedAttention(
            self.num_head, self.transformer_hidden, dropout=self.dropout
        )
        self.SublayerConnection = SublayerConnection(
            self.transformer_hidden, dropout=self.dropout
        )

        self.PositionwiseFeedForward = PositionwiseFeedForward(
            self.transformer_hidden, self.pe_hidden, dropout=0.1
        )

        if self.static_dim > 0:
            self.demo_proj_main = nn.Linear(self.static_dim, self.hidden_dim)

        self.dropout = nn.Dropout(p=self.dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def concare_encoder(self, input, static=None, mask=None):
        # input shape [batch_size, timestep, feature_dim]

        if self.static_dim > 0:
            demo_main = self.tanh(self.demo_proj_main(static)).unsqueeze(
                1
            )  # b hidden_dim

        batch_size = input.size(0)
        time_step = input.size(1)
        feature_dim = input.size(2)

        if self.transformer_hidden % self.num_head != 0:
            raise ValueError("transformer_hidden must be divisible by num_head")

        # forward
        GRU_embeded_input = self.GRUs[0](
            input[:, :, 0].unsqueeze(-1).to(device=input.device),
            torch.zeros(batch_size, self.hidden_dim)
            .to(device=input.device)
            .unsqueeze(0),
        )[
            0
        ]  # b t h
        Attention_embeded_input = self.LastStepAttentions[0](
            GRU_embeded_input, mask, input.device
        )[0].unsqueeze(
            1
        )  # b 1 h

        for i in range(feature_dim - 1):
            embeded_input = self.GRUs[i + 1](
                input[:, :, i + 1].unsqueeze(-1),
                torch.zeros(batch_size, self.hidden_dim)
                .to(device=input.device)
                .unsqueeze(0),
            )[
                0
            ]  # b 1 h
            embeded_input = self.LastStepAttentions[i + 1](
                embeded_input, mask, input.device
            )[0].unsqueeze(
                1
            )  # b 1 h
            Attention_embeded_input = torch.cat(
                (Attention_embeded_input, embeded_input), 1
            )  # b i h

        if self.static_dim > 0:
            Attention_embeded_input = torch.cat(
                (Attention_embeded_input, demo_main), 1
            )  # b i+1 h
        posi_input = self.dropout(
            Attention_embeded_input
        )  # batch_size * d_input+1 * hidden_dim

        contexts = self.SublayerConnection(
            posi_input,
            lambda x: self.MultiHeadedAttention(
                posi_input, posi_input, posi_input, None
            ),
        )  # # batch_size * d_input * hidden_dim

        DeCov_loss = contexts[1]
        contexts = contexts[0]

        contexts = self.SublayerConnection(
            contexts, lambda x: self.PositionwiseFeedForward(contexts)
        )[0]

        weighted_contexts, a = self.FinalAttentionQKV(contexts)
        return weighted_contexts, DeCov_loss

    def forward(
        self,
        x: torch.tensor,
        static: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor]:

        # rnn will only apply dropout between layers
        batch_size, time_steps, _ = x.size()
        out = torch.zeros((batch_size, time_steps, self.hidden_dim))
        out, decov = self.concare_encoder(x, static, mask)
        out = self.dropout(out)
        return out, decov
    

def random_init(dataset, num_centers, device):
    num_points = dataset.size(0)
    dimension = dataset.size(1)

    indices = torch.tensor(
        np.array(random.sample(range(num_points), k=num_centers)), dtype=torch.long
    )

    centers = torch.gather(
        dataset, 0, indices.view(-1, 1).expand(-1, dimension).to(device=device)
    )
    return centers


def compute_codes(dataset, centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)

    # print("size:", dataset.size(), centers.size())
    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(5e8 / num_centers)
    codes = torch.zeros(num_points, dtype=torch.long)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers**2, dim=1).view(1, -1)
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece**2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
    return codes


def update_centers(dataset, codes, num_centers, device):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    centers = torch.zeros(num_centers, dimension, dtype=torch.float).to(device=device)
    cnt = torch.zeros(num_centers, dtype=torch.float)
    centers.scatter_add_(
        0, codes.view(-1, 1).expand(-1, dimension).to(device=device), dataset
    )
    cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float))
    # Avoiding division by zero
    # Not necessary if there are no duplicates among the data points
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(num_centers, dtype=torch.float))
    centers /= cnt.view(-1, 1).to(device=device)
    return centers


def cluster(dataset, num_centers, device):
    centers = random_init(dataset, num_centers, device)
    codes = compute_codes(dataset, centers)
    num_iterations = 0
    while True:
        num_iterations += 1
        centers = update_centers(dataset, codes, num_centers, device)
        new_codes = compute_codes(dataset, centers)
        # Waiting until the clustering stops updating altogether
        # This is too strict in practice
        if torch.equal(codes, new_codes):
            break
        if num_iterations > 1000:
            break
        codes = new_codes
    return centers, codes


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features).float())
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).float())
        else:
            self.register_parameter("bias", None)
        self.initialize_parameters()

    def initialize_parameters(self):
        std = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, adj, x, device):
        y = torch.mm(x.float(), self.weight.float())
        output = torch.mm(adj.float(), y.float())
        if self.bias is not None:
            return output + self.bias.float().to(device=device)
        else:
            return output


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class GRASPLayer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        hidden_dim: int = 128,
        cluster_num: int = 2,
        dropout: int = 0.5,
        block: str = "ConCare",
    ):
        super(GRASPLayer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cluster_num = cluster_num
        self.dropout = dropout
        self.block = block

        if self.block == "ConCare":
            self.backbone = ConCareLayer(
                input_dim, static_dim, hidden_dim, hidden_dim, dropout=0
            )
        elif self.block == "GRU":
            self.backbone = RNNLayer(input_dim, hidden_dim, rnn_type="GRU", dropout=0)
        elif self.block == "LSTM":
            self.backbone = RNNLayer(input_dim, hidden_dim, rnn_type="LSTM", dropout=0)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.weight1 = nn.Linear(self.hidden_dim, 1)
        self.weight2 = nn.Linear(self.hidden_dim, 1)
        self.GCN = GraphConvolution(self.hidden_dim, self.hidden_dim, bias=True)
        self.GCN.initialize_parameters()
        self.GCN_2 = GraphConvolution(self.hidden_dim, self.hidden_dim, bias=True)
        self.GCN_2.initialize_parameters()
        self.A_mat = None

        self.bn = nn.BatchNorm1d(self.hidden_dim)

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)

        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, device):
        y = logits + self.sample_gumbel(logits.size()).to(device=device)
        return torch.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, device, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature, device)

        if not hard:
            return y.view(-1, self.cluster_num)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def grasp_encoder(self, input, static=None, mask=None):

        if self.block == "ConCare":
            hidden_t, _ = self.backbone(input, mask=mask, static=static)
        else:
            _, hidden_t = self.backbone(input, mask)
        hidden_t = torch.squeeze(hidden_t, 0)

        centers, codes = cluster(hidden_t, self.cluster_num, input.device)

        if self.A_mat == None:
            A_mat = np.eye(self.cluster_num)
        else:
            A_mat = kneighbors_graph(
                np.array(centers.detach().cpu().numpy()),
                20,
                mode="connectivity",
                include_self=False,
            ).toarray()

        adj_mat = torch.tensor(A_mat).to(device=input.device)

        e = self.relu(torch.matmul(hidden_t, centers.transpose(0, 1)))  # b clu_num

        scores = self.gumbel_softmax(e, temperature=1, device=input.device, hard=True)
        digits = torch.argmax(scores, dim=-1)  #  b

        h_prime = self.relu(self.GCN(adj_mat, centers, input.device))
        h_prime = self.relu(self.GCN_2(adj_mat, h_prime, input.device))

        clu_appendix = torch.matmul(scores, h_prime)

        weight1 = torch.sigmoid(self.weight1(clu_appendix))
        weight2 = torch.sigmoid(self.weight2(hidden_t))
        weight1 = weight1 / (weight1 + weight2)
        weight2 = 1 - weight1

        final_h = weight1 * clu_appendix + weight2 * hidden_t
        out = final_h
        
        return out

    def forward(
        self,
        x: torch.tensor,
        static: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
    ) -> torch.tensor:

        out = self.grasp_encoder(x, static, mask)
        out = self.dropout(out)

        return out


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
