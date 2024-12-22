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
from torch_scatter import scatter, scatter_add, scatter_mean, scatter_max, scatter_min
from torch_geometric.typing import (Adj, OptPairTensor, OptTensor, Size)
from torch import Tensor
from typing import Callable, Optional, Union, Tuple
from pyhealth.models import RNNLayer
from pyhealth.models.utils import get_last_visit
from sklearn.neighbors import kneighbors_graph
import dgl
import dgl.function as fn

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


########## KerPrint ##########


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=256, num_heads=4, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads)**-0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=256, ffn_dim=1024, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class EncoderLayer(nn.Module):

    def __init__(self, model_dim=256, num_heads=4, ffn_dim=1024, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):

        super(PositionalEncoding, self).__init__()

        self.max_seq_len = max_seq_len
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)] for pos in range(self.max_seq_len)
        ])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding.astype(np.float32))

        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        self.position_encoding = nn.Embedding(self.max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, input_len, device):
        # max_len = torch.max(input_len)
        # tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor

        pos = np.zeros([len(input_len), self.max_seq_len])
        for ind, length in enumerate(input_len):
            for pos_ind in range(1, length + 1):
                pos[ind, pos_ind - 1] = pos_ind
        input_pos = torch.LongTensor(pos).to(device)
        return self.position_encoding(input_pos), input_pos


def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


class EncoderNew(nn.Module):

    def __init__(self,
                 hita_input_size,
                 max_seq_len,
                 move_num,
                 hita_time_selection_layer_encoder,
                 hita_encoder_head_num,
                 hita_encoder_ffn_size,
                 hita_encoder_layer,
                 dropout=0.0):

        super(EncoderNew, self).__init__()

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(hita_input_size, hita_encoder_head_num, hita_encoder_ffn_size, dropout)
            for _ in range(hita_encoder_layer)
        ])

        self.bias_embedding = torch.nn.Parameter(torch.Tensor(hita_input_size))
        bound = 1 / math.sqrt(move_num)
        nn.init.uniform_(self.bias_embedding, -bound, bound)

        # self.weight_layer = torch.nn.Linear(model_dim, 1)
        self.pos_embedding = PositionalEncoding(hita_input_size, max_seq_len)
        self.hita_time_selection_layer_encoder = hita_time_selection_layer_encoder
        self.time_layer = torch.nn.Linear(self.hita_time_selection_layer_encoder[0],
                                          self.hita_time_selection_layer_encoder[1])
        self.selection_layer = torch.nn.Linear(1, self.hita_time_selection_layer_encoder[0])
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, sequence_embedding, seq_time_step, lengths, device):

        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature)
        output = sequence_embedding + self.bias_embedding
        output += time_feature
        output_pos, ind_pos = self.pos_embedding(lengths.unsqueeze(1), device)
        # print(output_pos)
        # print(output_pos.shape)
        output += output_pos
        self_attention_mask = padding_mask(ind_pos, ind_pos)

        attentions = []
        outputs = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)  #attention size(ffn_size,max_seq_len,max_seq_len)
            attentions.append(attention)
            outputs.append(output)
        # weight = torch.softmax(self.weight_layer(outputs[-1]), dim=1)
        # weight = weight * mask - 255 * (1 - mask)
        return output


class TransformerTime(nn.Module):

    def __init__(self, hita_input_size, hidden_dim, ff_dim, max_seq_len, move_num, hita_time_selection_layer_global,
                 hita_time_selection_layer_encoder, hita_encoder_head_num, hita_encoder_layers):

        super(TransformerTime, self).__init__()

        # self.prior_encoder = PriorEncoder(batch_size, options)
        self.time_encoder = TimeEncoder(hita_time_selection_layer_global)
        self.feature_encoder = EncoderNew(hita_input_size=hita_input_size,
                                          max_seq_len=max_seq_len,
                                          move_num=move_num,
                                          hita_time_selection_layer_encoder=hita_time_selection_layer_encoder,
                                          hita_encoder_head_num=hita_encoder_head_num,
                                          hita_encoder_ffn_size=ff_dim,
                                          hita_encoder_layer=hita_encoder_layers)
        self.self_layer = torch.nn.Linear(hita_input_size, 1)
        # self.classify_layer = torch.nn.Linear(256, 2)
        self.quiry_layer = torch.nn.Linear(hita_input_size, hidden_dim)
        self.quiry_weight_layer = torch.nn.Linear(hita_input_size, 2)
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = 0.5
        self.dropout = nn.Dropout(dropout_rate)

    def get_self_attention(self, features, query, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill(mask, -np.inf), dim=1)
        # attention = torch.sum(key * query, 2, keepdim=True) / 8
        return attention

    def forward(self, sequence_embedding, seq_time_step, mask_final, lengths, device):
        # sequence_embedding: [batch_size, length, embedding_size]
        # seq_time_step: [batch_size, length] the day times to the final visit
        # features size:(batch_size, max_seq_len, hita_input_size)
        # mask_final size:(batch_size, max_seq_len, 1)

        features = self.feature_encoder(sequence_embedding, seq_time_step, lengths, device)
        final_statues = features * mask_final
        print(final_statues.shape)
        final_statues = final_statues.sum(dim=1, keepdim=False)
        print(final_statues.shape)
        quiryes = self.relu(self.quiry_layer(final_statues))  #(batch_size,1,global_qurey_size)
        print(quiryes.shape)
        exit()

        self_weight = self.get_self_attention(features, quiryes, None)  #(batch_size,max_visit_len,1) # no global
        time_weight = self.time_encoder(seq_time_step, quiryes, None)  #(batch_size,max_visit_len,1)
        # self_weight = self.get_self_attention(features, quiryes, mask_mult)  #(batch_size,max_visit_len,1) # no global
        # time_weight = self.time_encoder(seq_time_step, quiryes, mask_mult)  #(batch_size,max_visit_len,1)
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2)

        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)

        return averaged_features, self_weight


class TimeEncoder(nn.Module):

    def __init__(self, hita_time_selection_layer_global):
        super(TimeEncoder, self).__init__()
        self.hita_time_selection_layer_global = hita_time_selection_layer_global
        self.selection_layer = torch.nn.Linear(1, self.hita_time_selection_layer_global[0])
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight_layer = torch.nn.Linear(self.hita_time_selection_layer_global[0],
                                            self.hita_time_selection_layer_global[1])

    def forward(self, seq_time_step, final_queries, mask):
        selection_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        selection_feature = self.relu(self.weight_layer(selection_feature))
        selection_feature = torch.sum(selection_feature * final_queries, 2,
                                      keepdim=True) / (self.hita_time_selection_layer_global[1]**0.5)
        selection_feature = selection_feature.masked_fill_(mask, -np.inf)
        # time_weights = self.weight_layer(selection_feature)
        return torch.softmax(selection_feature, 1)


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout):
        super(Aggregator, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.message_dropout = nn.Dropout(dropout)
        self.W = nn.Linear(self.in_dim, self.out_dim)
        self.activation = nn.LeakyReLU()

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    def forward(self, x, edge_index, edge_attr):

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.activation(self.W(out))
        out = self.message_dropout(out)

        return out

    def message(self, x_j, edge_attr):
        return x_j * edge_attr.unsqueeze(-1)

    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=0, reduce="add")

    def update(self, aggr_out):
        return aggr_out


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
class CompGCNConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 act=lambda x: x,
                 bias=True,
                 drop_rate=0.,
                 opn='rotate',
                 num_base=-1,
                 num_rel=None):

        super(CompGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act  # activation function
        self.device = None
        self.rel = None
        self.opn = opn

        # relation-type specific parameter
        self.in_w = self.get_param([in_channels, out_channels])
        self.out_w = self.get_param([in_channels, out_channels])
        self.loop_w = self.get_param([in_channels, out_channels])
        self.w_rel = self.get_param([in_channels, out_channels])  # transform embedding of relations to next layer
        self.loop_rel = self.get_param([1, in_channels])  # self-loop embedding

        self.drop = nn.Dropout(drop_rate)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        if num_base > 0:
            self.rel_wt = self.get_param([num_rel * 2, num_base])
        else:
            self.rel_wt = None

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def message_func(self, edges):
        edge_type = edges.data['type']  # [E, 1]
        edge_num = edge_type.shape[0]
        edge_data = self.comp(edges.src['h'], self.rel[edge_type])  # [E, in_channel]
        # msg = torch.bmm(edge_data.unsqueeze(1),
        #                 self.w[edge_dir.squeeze()]).squeeze()  # [E, 1, in_c] @ [E, in_c, out_c]
        # msg = torch.bmm(edge_data.unsqueeze(1),
        #                 self.w.index_select(0, edge_dir.squeeze())).squeeze()  # [E, 1, in_c] @ [E, in_c, out_c]
        # NOTE: first half edges are all in-directions, last half edges are out-directions.
        msg = torch.cat([torch.matmul(edge_data[:edge_num // 2, :], self.in_w),
                         torch.matmul(edge_data[edge_num // 2:, :], self.out_w)])
        msg = msg * edges.data['norm'].reshape(-1, 1)  # [E, D] * [E, 1]
        return {'msg': msg}

    def reduce_func(self, nodes):
        return {'h': self.drop(nodes.data['h']) / 3}

    def comp(self, h, edge_data):

        def com_mult(a, b):
            r1, i1 = a[..., 0], a[..., 1]
            r2, i2 = b[..., 0], b[..., 1]
            return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

        def conj(a):
            a[..., 1] = -a[..., 1]
            return a

        def ccorr(a, b):
            # Compute the Fourier transforms
            a_fft = torch.fft.rfft(a, dim=-1)
            b_fft = torch.fft.rfft(b, dim=-1)

            # Compute the conjugate of the Fourier transform of `a`
            a_fft_conj = torch.conj(a_fft)

            # Perform element-wise multiplication in the Fourier domain
            result_fft = a_fft_conj * b_fft

            # Inverse Fourier transform to get the result in the time domain
            result = torch.fft.irfft(result_fft, n=a.shape[-1], dim=-1)

            return result
        
        def rotate(h, r):
            d = h.shape[-1]
            h_re, h_im = torch.split(h, d // 2, -1)
            r_re, r_im = torch.split(r, d // 2, -1)
            return torch.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1)

        if self.opn == 'mult':
            return h * edge_data
        elif self.opn == 'sub':
            return h - edge_data
        elif self.opn == 'corr':
            return ccorr(h, edge_data.expand_as(h))
        elif self.opn == 'rotate':
            return rotate(h, edge_data.expand_as(h))
        else:
            raise KeyError(f'composition operator {self.opn} not recognized.')

    def forward(self, g: dgl.DGLGraph, x, rel_repr, edge_type, edge_norm):
        """
        :param g: dgl Graph, a graph without self-loop
        :param x: input node features, [V, in_channel]
        :param rel_repr: input relation features: 1. not using bases: [num_rel*2, in_channel]
                                                  2. using bases: [num_base, in_channel]
        :param edge_type: edge type, [E]
        :param edge_norm: edge normalization, [E]
        :return: x: output node features: [V, out_channel]
                 rel: output relation features: [num_rel*2, out_channel]
        """

        self.device = x.device
        g = g.local_var()
        g.ndata['h'] = x
        g.edata['type'] = edge_type
        g.edata['norm'] = edge_norm

        if self.rel_wt is None:
            self.rel = rel_repr
        else:
            self.rel = torch.mm(self.rel_wt, rel_repr)  # [num_rel*2, num_base] @ [num_base, in_c]

        g.update_all(self.message_func, fn.sum(msg='msg', out='h'), self.reduce_func)
        x = g.ndata.pop('h') + torch.mm(self.comp(x, self.loop_rel), self.loop_w) / 3

        if self.bias is not None:
            x = x + self.bias

        x = self.bn(x)

        return self.act(x), torch.matmul(self.rel, self.w_rel)
    

class PrototypeLearning(nn.Module):
    def __init__(self,
                 prototype_num,
                 hidden_dim,
                 commitment_cost=None,
                 attn_iter=None,
                 learning_mode='VQVAE',
                 prediction_mode=None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.prototype_num = prototype_num
        self.learning_mode = learning_mode
        self.prediction_mode = prediction_mode
        
        self.prototypes = nn.Parameter(torch.Tensor(self.prototype_num, self.hidden_dim))
        self.prototypes.data.uniform_(-1 / self.prototype_num, 1 / self.prototype_num)

        if self.learning_mode == 'VQVAE':
            self.commitment_cost = commitment_cost

        elif self.learning_mode == 'QKV':
            self.attn_iter = attn_iter 
            self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim,
                                              num_heads=1,
                                              dropout=0.,
                                              batch_first=True)

    def forward(self, x, mask=None):
        if self.learning_mode == 'VQVAE':
            # Calculate distances
            distances = (torch.sum(x**2, dim=1, keepdim=True) 
                        + torch.sum(self.prototypes**2, dim=1)
                        - 2 * torch.matmul(x, self.prototypes.t()))
                
            # Encoding
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(encoding_indices.shape[0], self.prototype_num, device=x.device)
            encodings.scatter_(1, encoding_indices, 1)
            
            # Quantize and unflatten
            quantized = torch.matmul(encodings, self.prototypes)
            
            # Loss
            e_latent_loss = F.mse_loss(quantized.detach(), x)
            q_latent_loss = F.mse_loss(quantized, x.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
            
            if self.prediction_mode == 'onehot':
                quantized = x + (quantized - x).detach()
                return quantized, loss, self.prototypes
            
            return None, loss, self.prototypes
        
        elif self.learning_mode == 'QKV':
            prototypes = self.prototypes

            for _ in range(self.attn_iter):
                prototypes, _ = self.attn(prototypes, x, x, key_padding_mask=mask)

            return None, None, prototypes

        else:
            raise ValueError('Select a suitable learning_mode')


class PrototypePrediction(nn.Module):
    def __init__(self, prototype_num, hidden_dim, out_dim, prediction_mode='simC', learning_mode='QKV'):
        super().__init__()

        self.prototype_num = prototype_num
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.prediction_mode = prediction_mode
        self.learning_mode = learning_mode
        
        if self.prediction_mode == 'simH':
            self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
            self.prototype_heads = nn.ModuleList()
            for _ in range(self.prototype_num):
                self.prototype_heads.append(nn.Linear(self.hidden_dim, self.out_dim))
        
        elif self.prediction_mode == 'simC':
            # self.sim_alpha = nn.Parameter(torch.ones(1))
            # self.sim_beta = nn.Parameter(torch.zeros(1))
            self.projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def get_simH_prob(self, x, prototypes):

        def sinkhorn(out, epsilon=0.05):
            Q = torch.exp(out / epsilon)
            sum_Q = torch.sum(Q, dim=1, keepdim=True) 
            Q = Q / (sum_Q)
            return Q

        prob = torch.mm(self.l2norm(x), self.l2norm(prototypes).t())
        prob = sinkhorn(prob)

        return prob

    def get_simC_prob(self, x, prototypes):
        def pairwise_cos_sim(x1: torch.Tensor, x2:torch.Tensor):
            x1 = F.normalize(x1,dim=-1)
            x2 = F.normalize(x2,dim=-1)
            sim = torch.matmul(x1, x2.transpose(-2, -1))
            return sim
        
        prob = pairwise_cos_sim(x, prototypes)
        prob = torch.softmax(prob, dim=-1)
        return prob
    
    def get_onehot_prob(self, x, prototypes):
        prob = self.get_simC_prob(x, prototypes)
        _, max_idx = prob.max(dim=1, keepdim=True)
        mask = torch.zeros_like(prob)
        mask.scatter_(1, max_idx, 1.)
        prob = prob * mask
        return prob

    def forward(self, x, prototypes):
        '''
        x: (B, patient_num, D)
        prototypes: (K, D)
        '''

        if self.prediction_mode == 'simH':
            prob = self.get_simH_prob(x, prototypes)    # (B, K)

            output = []
            for layer in self.prototype_heads:
                output.append(layer(x))

            output = torch.stack(output, dim=-1).to(x.device)   # (B, O, K)
            prob = prob.unsqueeze(-1)           # (B, K, 1)
            x = torch.matmul(output, prob).squeeze(-1)  # (B, O)

        elif self.prediction_mode == 'simC':
            prob = self.get_simC_prob(x, prototypes)    # (B, K)
            x_prototype = torch.matmul(prob, prototypes) / self.prototype_num
            x = self.projection(x_prototype)

        elif self.learning_mode != 'VQVAE' and self.prediction_mode == 'onehot':
            prob = self.get_onehot_prob(x, prototypes)
            x = torch.matmul(prob, prototypes) / self.prototype_num

        else:
            raise ValueError('Select a suitable prediction_mode')

        return x
