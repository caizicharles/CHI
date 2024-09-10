import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GATConv, GINConv, global_mean_pool
from typing import Dict

from .modules import CodeTypeEmbedding, DataEmbedding, CompGCNConv, SetTransformer, ImportanceAttention,\
BiAttentionGNNConv, CausalConv1d, Recalibration, DeeprLayer, TimeGapEmbedding, GRASPLayer


class GRU(nn.Module):

    def __init__(self, args):
        super().__init__()

        device = args['device']
        self.num_nodes = args['num_nodes']
        self.visit_thresh = args['visit_thresh']
        self.visit_code_num = args['visit_code_num']
        self.input_dim = args['input_dim']
        self.hidden_dim = args['hidden_dim']
        self.num_layers = args['layers']
        self.out_dim = args['out_dim']
        node_embed = args['global_node_attr']

        self.full_node_ids = torch.arange(self.num_nodes + 1).to(device)

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes + 1, self.input_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)

        self.h_0 = nn.Parameter(torch.Tensor(self.num_layers, 1, self.hidden_dim))
        nn.init.xavier_uniform_(self.h_0)

        self.gru = nn.GRU(input_size=self.visit_code_num * self.input_dim,
                          hidden_size=self.hidden_dim,
                          num_layers=self.num_layers,
                          batch_first=True)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, visit_order, attn_mask, **kwargs):

        x = self.node_embed(self.full_node_ids)

        B, V, N = node_ids.shape
        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.input_dim))
        x = x.reshape(B, V, N * self.input_dim)  # (B,V,N*D)

        x, h = self.gru(x, self.h_0.repeat(1, B, 1))  # (B,V,D), (layer_num,B,D)

        visit_mask = attn_mask.all(dim=-1)
        visit_mask = ~visit_mask
        visit_mask = visit_mask.to(int)
        mask = torch.sum(visit_mask, dim=-1) - 1
        x = x[torch.arange(B), mask]  # (B,D)

        out = self.head(x)

        return {'logits': [out], 'prototypes': None, 'embeddings': None, 'scores': None}


class Transformer(nn.Module):

    def __init__(self, args):
        super().__init__()

        device = args['device']
        self.num_nodes = args['num_nodes']
        self.visit_thresh = args['visit_thresh']
        self.visit_code_num = args['visit_code_num']
        self.start_embed_dim = args['start_embed_dim']
        self.input_dim = args['input_dim']
        self.ff_dim = args['ff_dim']
        self.head_num = args['head_num']
        self.encoder_depth = args['encoder_depth']
        self.decoder_depth = args['decoder_depth']
        self.out_dim = args['out_dim']
        node_embed = args['global_node_attr']

        self.full_node_ids = torch.arange(self.num_nodes + 1).to(device)

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes + 1, self.input_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)

        self.fc = nn.Linear(self.visit_code_num * self.start_embed_dim, self.input_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.visit_code_num * self.input_dim,
                                       nhead=self.head_num,
                                       dim_feedforward=self.ff_dim,
                                       dropout=0.,
                                       batch_first=True), self.encoder_depth)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.visit_code_num * self.input_dim,
                                       nhead=self.head_num,
                                       dim_feedforward=self.ff_dim,
                                       dropout=0.,
                                       batch_first=True), self.decoder_depth)
        self.head = nn.Linear(self.input_dim, self.out_dim)

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, visit_order, attn_mask, **kwargs):

        B, V, N = node_ids.shape
        x = self.node_embed(self.full_node_ids)

        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.input_dim))
        x = x.reshape(B, V, N * self.input_dim)  # (B,V,N*D)

        visit_mask = attn_mask.all(dim=-1)

        x = self.encoder(x, src_key_padding_mask=visit_mask)
        x = self.decoder(x, x, tgt_key_padding_mask=visit_mask)
        x = self.fc(x)  # (B,V,D)

        visit_mask = ~visit_mask
        mask = visit_mask.to(int).unsqueeze(-1)
        x = x * mask
        x = x.sum(dim=1)  # (B,D)

        count = mask.sum(dim=1)
        count = count.clamp(min=1)
        x = x / count  # (B,D)

        out = self.head(x)

        return {'logits': [out], 'prototypes': None, 'embeddings': None, 'scores': None}


class Deepr(nn.Module):

    def __init__(self, args):
        super().__init__()

        device = args['device']
        self.num_nodes = args['num_nodes']
        self.visit_thresh = args['visit_thresh']
        self.max_weeks_between = args['max_weeks_between']
        self.visit_code_num = args['visit_code_num']
        self.start_embed_dim = args['start_embed_dim']
        self.hidden_dim = args['hidden_dim']
        self.window_size = args['window_size']
        self.out_dim = args['out_dim']
        node_embed = args['global_node_attr']

        self.full_node_ids = torch.arange(self.num_nodes + 1).to(device)

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes + 1, self.start_embed_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)

        self.time_gap_embed = TimeGapEmbedding(self.start_embed_dim, device)

        self.layer = DeeprLayer(feature_size=self.start_embed_dim, window=self.window_size, hidden_size=self.hidden_dim)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, visit_order, attn_mask, **kwargs):

        B, V, N = node_ids.shape
        x = self.node_embed(self.full_node_ids)

        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.start_embed_dim))  # (B,V,N,D)

        gap_embed = self.time_gap_embed(visit_times)
        first_row = gap_embed[:, 0, :]
        rest_rows = gap_embed[:, 1:, :]
        gap_embed = torch.cat((rest_rows, first_row.unsqueeze(1)), dim=1)
        gap_embed = gap_embed.unsqueeze(-2)

        x = torch.cat((x, gap_embed), dim=-2)

        visit_mask = attn_mask.all(dim=-1)
        visit_mask = ~visit_mask
        visit_mask = visit_mask.unsqueeze(-1).unsqueeze(-1)
        x = self.layer(x, visit_mask.float())  # (B,D)

        out = self.head(x)

        return {'logits': [out], 'prototypes': None, 'embeddings': None, 'scores': None}


class AdaCare(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.device = args['device']
        self.num_nodes = args['num_nodes']
        self.visit_thresh = args['visit_thresh']
        self.visit_code_num = args['visit_code_num']
        self.input_dim = args['input_dim']
        self.hidden_dim = args['hidden_dim']
        self.kernel_size = args['kernel_size']
        self.kernel_num = args['kernel_num']
        self.out_dim = args['out_dim']
        self.r_v = args['r_v']
        self.r_c = args['r_c']
        self.activation = args['activation']
        node_embed = args['global_node_attr']
        self.dropout = 0.

        self.full_node_ids = torch.arange(self.num_nodes + 1).to(self.device)

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes + 1, self.start_embed_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)

        self.nn_conv1 = CausalConv1d(self.visit_code_num * self.input_dim, self.kernel_num, self.kernel_size, 1, 1)
        self.nn_conv3 = CausalConv1d(self.visit_code_num * self.input_dim, self.kernel_num, self.kernel_size, 1, 3)
        self.nn_conv5 = CausalConv1d(self.visit_code_num * self.input_dim, self.kernel_num, self.kernel_size, 1, 5)
        torch.nn.init.xavier_uniform_(self.nn_conv1.weight)
        torch.nn.init.xavier_uniform_(self.nn_conv3.weight)
        torch.nn.init.xavier_uniform_(self.nn_conv5.weight)

        self.nn_convse = Recalibration(3 * self.kernel_num, self.r_c, use_h=False, use_c=True, activation='sigmoid')
        self.nn_inputse = Recalibration(self.visit_code_num * self.input_dim,
                                        self.r_v,
                                        use_h=False,
                                        use_c=True,
                                        activation=self.activation)
        self.rnn = nn.GRUCell(self.visit_code_num * self.input_dim + 3 * self.kernel_num, self.hidden_dim)
        self.nn_output = nn.Linear(self.hidden_dim, self.out_dim)
        self.nn_dropout = nn.Dropout(self.dropout)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, visit_order, attn_mask, **kwargs):

        B, V, N = node_ids.shape
        input = self.node_embed(self.full_node_ids)
        input = input.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        input = torch.gather(input, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.input_dim))
        input = input.reshape(B, V, N * self.input_dim)  # (B,V,N*D)

        cur_h = Variable(torch.zeros(B, self.hidden_dim)).to(self.device)
        inputse_att = []
        convse_att = []
        h = []

        conv_input = input.permute(0, 2, 1)
        conv_res1 = self.nn_conv1(conv_input)  # (B,kernel_num,V)
        conv_res3 = self.nn_conv3(conv_input)  # (B,kernel_num,V)
        conv_res5 = self.nn_conv5(conv_input)  # (B,kernel_num,V)

        conv_res = torch.cat((conv_res1, conv_res3, conv_res5), dim=1)
        conv_res = self.relu(conv_res)

        for cur_time in range(V):
            convse_res, cur_convatt = self.nn_convse(conv_res[:, :, :cur_time + 1], device=self.device)

            inputse_res, cur_inputatt = self.nn_inputse(input[:, :cur_time + 1, :].permute(0, 2, 1), device=self.device)

            cur_input = torch.cat((convse_res[:, :, -1], inputse_res[:, :, -1]), dim=-1)
            cur_h = self.rnn(cur_input, cur_h)

            h.append(cur_h)
            convse_att.append(cur_convatt)
            inputse_att.append(cur_inputatt)

        h = torch.stack(h).permute(1, 0, 2)
        h_reshape = h.contiguous().view(B * V, self.hidden_dim)
        if self.dropout > 0.0:
            h_reshape = self.nn_dropout(h_reshape)
        output = self.nn_output(h_reshape)
        # output = self.sigmoid(output)
        output = output.contiguous().view(B, V, self.out_dim)

        visit_mask = attn_mask.all(dim=-1)
        visit_mask = ~visit_mask
        visit_mask = visit_mask.to(int)
        mask = torch.sum(visit_mask, dim=-1) - 1
        output = output[torch.arange(B), mask]

        return {'logits': [output], 'prototypes': None, 'embeddings': None, 'scores': None}


class StageNet(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.device = args['device']
        self.num_nodes = args['num_nodes']
        self.visit_thresh = args['visit_thresh']
        self.visit_code_num = args['visit_code_num']
        self.input_dim = args['input_dim']
        self.hidden_dim = args['hidden_dim']
        self.conv_dim = args['hidden_dim']
        self.conv_size = args['conv_size']
        self.output_dim = args['out_dim']
        self.levels = args['levels']
        node_embed = args['global_node_attr']
        self.dropconnect = 0.
        self.dropout = 0.
        self.dropres = 0.
        self.chunk_size = self.hidden_dim // self.levels
        assert self.hidden_dim % self.levels == 0

        self.full_node_ids = torch.arange(self.num_nodes + 1).to(self.device)

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes + 1, self.start_embed_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)

        self.kernel = nn.Linear(int(self.input_dim * self.visit_code_num + 1),
                                int(self.hidden_dim * 4 + self.levels * 2))
        nn.init.xavier_uniform_(self.kernel.weight)
        nn.init.zeros_(self.kernel.bias)
        self.recurrent_kernel = nn.Linear(int(self.hidden_dim + 1), int(self.hidden_dim * 4 + self.levels * 2))
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.recurrent_kernel.bias)

        self.nn_scale = nn.Linear(int(self.hidden_dim), int(self.hidden_dim // 6))
        self.nn_rescale = nn.Linear(int(self.hidden_dim // 6), int(self.hidden_dim))
        self.nn_conv = nn.Conv1d(int(self.hidden_dim), int(self.conv_dim), int(self.conv_size), 1)
        self.nn_output = nn.Linear(int(self.conv_dim), int(self.output_dim))

        if self.dropconnect:
            self.nn_dropconnect = nn.Dropout(p=self.dropconnect)
            self.nn_dropconnect_r = nn.Dropout(p=self.dropconnect)
        if self.dropout:
            self.nn_dropout = nn.Dropout(p=self.dropout)
            self.nn_dropres = nn.Dropout(p=self.dropres)

    def cumax(self, x, mode='l2r'):
        if mode == 'l2r':
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return x
        elif mode == 'r2l':
            x = torch.flip(x, [-1])
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return torch.flip(x, [-1])
        else:
            return x

    def step(self, inputs, c_last, h_last, interval):
        x_in = inputs  # (B,ND)

        # Integrate inter-visit time intervals
        interval = interval.unsqueeze(-1)
        x_out1 = self.kernel(torch.cat((x_in, interval), dim=-1))
        x_out2 = self.recurrent_kernel(torch.cat((h_last, interval), dim=-1))

        if self.dropconnect:
            x_out1 = self.nn_dropconnect(x_out1)
            x_out2 = self.nn_dropconnect_r(x_out2)

        x_out = x_out1 + x_out2
        f_master_gate = self.cumax(x_out[:, :self.levels], 'l2r')
        f_master_gate = f_master_gate.unsqueeze(2)
        i_master_gate = self.cumax(x_out[:, self.levels:self.levels * 2], 'r2l')
        i_master_gate = i_master_gate.unsqueeze(2)
        x_out = x_out[:, self.levels * 2:]
        x_out = x_out.reshape(-1, self.levels * 4, self.chunk_size)
        f_gate = torch.sigmoid(x_out[:, :self.levels])
        i_gate = torch.sigmoid(x_out[:, self.levels:self.levels * 2])
        o_gate = torch.sigmoid(x_out[:, self.levels * 2:self.levels * 3])
        c_in = torch.tanh(x_out[:, self.levels * 3:])
        c_last = c_last.reshape(-1, self.levels, self.chunk_size)
        overlap = f_master_gate * i_master_gate
        c_out = overlap * (f_gate * c_last + i_gate * c_in) + (f_master_gate - overlap) * c_last + (i_master_gate -
                                                                                                    overlap) * c_in
        h_out = o_gate * torch.tanh(c_out)
        c_out = c_out.reshape(-1, self.hidden_dim)
        h_out = h_out.reshape(-1, self.hidden_dim)
        out = torch.cat([h_out, f_master_gate[..., 0], i_master_gate[..., 0]], 1)
        return out, c_out, h_out

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, visit_order, attn_mask, **kwargs):

        B, V, N = node_ids.shape
        input = self.node_embed(self.full_node_ids)

        input = input.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        input = torch.gather(input, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.input_dim))
        input = input.reshape(B, V, N * self.input_dim)

        _, time_step, ND = input.size()
        c_out = torch.zeros(B, self.hidden_dim).to(self.device)
        h_out = torch.zeros(B, self.hidden_dim).to(self.device)
        tmp_h = torch.zeros_like(h_out, dtype=torch.float32).view(-1).repeat(self.conv_size).view(
            self.conv_size, B, self.hidden_dim).to(self.device)
        tmp_dis = torch.zeros((self.conv_size, B)).to(self.device)

        h = []
        origin_h = []
        distance = []
        for t in range(time_step):
            out, c_out, h_out = self.step(input[:, t, :], c_out, h_out, visit_times[:, t])
            cur_distance = 1 - torch.mean(out[..., self.hidden_dim:self.hidden_dim + self.levels], -1)
            cur_distance_in = torch.mean(out[..., self.hidden_dim + self.levels:], -1)
            origin_h.append(out[..., :self.hidden_dim])
            tmp_h = torch.cat((tmp_h[1:], out[..., :self.hidden_dim].unsqueeze(0)), 0)
            tmp_dis = torch.cat((tmp_dis[1:], cur_distance.unsqueeze(0)), 0)
            distance.append(cur_distance)

            # Re-weighted convolution operation
            local_dis = tmp_dis.permute(1, 0)
            local_dis = torch.cumsum(local_dis, dim=1)
            local_dis = torch.softmax(local_dis, dim=1)
            local_h = tmp_h.permute(1, 2, 0)
            local_h = local_h * local_dis.unsqueeze(1)

            # Re-calibrate Progression patterns
            local_theme = torch.mean(local_h, dim=-1)
            local_theme = self.nn_scale(local_theme)
            local_theme = torch.relu(local_theme)
            local_theme = self.nn_rescale(local_theme)
            local_theme = torch.sigmoid(local_theme)

            local_h = self.nn_conv(local_h).squeeze(-1)
            local_h = local_theme * local_h
            h.append(local_h)

        origin_h = torch.stack(origin_h).permute(1, 0, 2)
        rnn_outputs = torch.stack(h).permute(1, 0, 2)
        if self.dropres > 0.0:
            origin_h = self.nn_dropres(origin_h)
        rnn_outputs = rnn_outputs + origin_h
        rnn_outputs = rnn_outputs.contiguous().view(-1, rnn_outputs.size(-1))
        if self.dropout > 0.0:
            rnn_outputs = self.nn_dropout(rnn_outputs)

        # Prediction
        output = self.nn_output(rnn_outputs)
        output = output.contiguous().view(B, time_step, self.output_dim)

        visit_mask = attn_mask.all(dim=-1)
        visit_mask = ~visit_mask
        visit_mask = visit_mask.to(int)
        mask = torch.sum(visit_mask, dim=-1) - 1
        output = output[torch.arange(B), mask]

        return {'logits': [output], 'prototypes': None, 'embeddings': None, 'scores': None}


class GRASP(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.device = args['device']
        self.num_nodes = args['num_nodes']
        self.num_edges = args['num_edges']
        self.num_visits = args['visit_thresh']
        self.num_codes = args['visit_code_num']
        self.out_dim = args['out_dim']
        self.start_embed_dim = args['start_embed_dim']
        self.hidden_dim = args['hidden_dim']
        self.cluster_num = args['cluster_num']
        self.block = args['block']
        node_embed = args['global_node_attr']

        self.full_node_ids = torch.arange(self.num_nodes + 1).to(self.device)

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes + 1, self.start_embed_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)
            self.node_projection = nn.Linear(node_embed.size(-1), self.start_embed_dim)

        self.grasp = GRASPLayer(input_dim=self.hidden_dim,
                                static_dim=0,
                                hidden_dim=self.hidden_dim,
                                cluster_num=self.cluster_num,
                                block=self.block)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, visit_order, attn_mask, **kwargs):

        x = self.node_embed(self.full_node_ids)  # (node_num,D)
        x = self.node_projection(x)

        B, V, N = node_ids.shape
        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim))
        x = torch.sum(x, dim=2)

        visit_mask = attn_mask.all(dim=-1)
        visit_mask = ~visit_mask
        visit_mask = visit_mask.to(int)

        x = self.grasp(x, mask=visit_mask)  # (B,D)

        out = self.head(x)

        return {'logits': [out], 'prototypes': None, 'embeddings': None, 'scores': None}


class GraphCare(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.gnn = args['gnn']
        self.layer = args['layer']
        self.embedding_dim = args['start_embed_dim']
        self.decay_rate = args['decay_rate']
        self.patient_mode = args['patient_mode']
        self.use_alpha = args['use_alpha']
        self.use_beta = args['use_beta']
        self.edge_attn = args['edge_attn']
        self.num_nodes = args['num_nodes']
        self.num_edges = args['num_edges']
        self.max_visit = args['visit_thresh']
        self.hidden_dim = args['hidden_dim']
        self.out_channels = args['out_dim']
        node_embed = args['global_node_attr']
        edge_embed = args['global_edge_attr']
        self.drop_rate = 0.
        self.dropout = 0.5
        self.attn_init = 1
        self.self_attn = 0.

        j = torch.arange(self.max_visit).float()
        self.lambda_j = torch.exp(self.decay_rate * (self.max_visit - j)).unsqueeze(0).reshape(1, self.max_visit,
                                                                                               1).float()

        if node_embed is None:
            self.node_emb = nn.Embedding(self.num_nodes + 1, self.embedding_dim, padding_idx=0)
        else:
            self.node_emb = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)

        if edge_embed is None:
            self.edge_emb = nn.Embedding(self.num_edges + 1, self.embedding_dim, padding_idx=0)
        else:
            edge_emb = edge_embed.to(torch.float32)
            self.edge_emb = nn.Embedding.from_pretrained(edge_emb, freeze=False, padding_idx=0)

        self.lin = nn.Linear(node_embed.size(-1), self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)

        self.alpha_attn = nn.ModuleDict()
        self.beta_attn = nn.ModuleDict()
        self.conv = nn.ModuleDict()
        self.bn_gnn = nn.ModuleDict()

        self.leakyrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.tahh = nn.Tanh()

        for layer in range(1, self.layer + 1):
            if self.use_alpha:
                self.alpha_attn[str(layer)] = nn.Linear(self.num_nodes + 1, self.num_nodes + 1)

                if self.attn_init is not None:
                    # self.attn_init = self.attn_init.float()  # Convert attn_init to float
                    attn_init_matrix = torch.eye(
                        self.num_nodes + 1).float() * self.attn_init  # Multiply the identity matrix by attn_init
                    self.alpha_attn[str(layer)].weight.data.copy_(
                        attn_init_matrix)  # Copy the modified attn_init_matrix to the weights

                else:
                    nn.init.xavier_normal_(self.alpha_attn[str(layer)].weight)
            if self.use_beta:
                self.beta_attn[str(layer)] = nn.Linear(self.num_nodes + 1, 1)
                nn.init.xavier_normal_(self.beta_attn[str(layer)].weight)
            if self.gnn == "BAT":
                self.conv[str(layer)] = BiAttentionGNNConv(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                           edge_dim=self.hidden_dim,
                                                           edge_attn=self.edge_attn,
                                                           eps=self.self_attn)
            elif self.gnn == "GAT":
                self.conv[str(layer)] = GATConv(self.hidden_dim, self.hidden_dim)
            elif self.gnn == "GIN":
                self.conv[str(layer)] = GINConv(nn.Linear(self.hidden_dim, self.hidden_dim))

            # self.bn_gnn[str(layer)] = nn.BatchNorm1d(hidden_dim)

        if self.patient_mode == "joint":
            self.MLP = nn.Linear(self.hidden_dim * 2, self.out_channels)
        else:
            self.MLP = nn.Linear(self.hidden_dim, self.out_channels)

    def to(self, device):
        super().to(device)
        self.lambda_j = self.lambda_j.float().to(device)
        return self

    def forward(self,
                node_ids,
                edge_idx,
                edge_attr,
                visit_times,
                visit_order,
                attn_mask,
                in_drop=False,
                store_attn=False,
                **kwargs):

        node_ids = kwargs['cat_node_ids']
        edge_ids = kwargs['cat_edge_ids']
        edge_index = kwargs['cat_edge_index']
        visit_node = kwargs['visit_nodes']  # (B,V,node_num)
        ehr_nodes = kwargs['ehr_nodes']  # (B,node_num)
        batch = kwargs['batch']

        if in_drop and self.drop_rate > 0:
            edge_count = edge_index.size(1)
            edges_to_remove = int(edge_count * self.drop_rate)
            indices_to_remove = set(random.sample(range(edge_count), edges_to_remove))
            edge_index = edge_index[:,
                                    [i for i in range(edge_count) if i not in indices_to_remove]].to(edge_index.device)
            edge_ids = torch.tensor([rel_id for i, rel_id in enumerate(edge_ids) if i not in indices_to_remove],
                                    device=edge_ids.device)

        x = self.node_emb(node_ids).float()
        edge_attr = self.edge_emb(edge_ids).float()

        x = self.lin(x)
        edge_attr = self.lin(edge_attr)

        if store_attn:
            self.alpha_weights = []
            self.beta_weights = []
            self.attention_weights = []
            self.edge_weights = []

        for layer in range(1, self.layer + 1):
            if self.use_alpha:
                # alpha = masked_softmax((self.leakyrelu(self.alpha_attn[str(layer)](visit_node.float()))), mask=visit_node>1, dim=1)
                alpha = torch.softmax((self.alpha_attn[str(layer)](visit_node.float())),
                                      dim=1)  # (batch, max_visit, num_nodes)

            if self.use_beta:
                # beta = masked_softmax((self.leakyrelu(self.beta_attn[str(layer)](visit_node.float()))), mask=visit_node>1, dim=0) * self.lambda_j
                beta = torch.tanh(
                    (self.beta_attn[str(layer)](visit_node.float()))) * self.lambda_j  # (batch, max_visit, 1)

            if self.use_alpha and self.use_beta:
                attn = alpha * beta  # (batch, max_visit, num_nodes)
            elif self.use_alpha:
                attn = alpha * torch.ones((batch.max().item() + 1, self.max_visit, 1)).to(edge_index.device)
            elif self.use_beta:
                attn = beta * torch.ones((batch.max().item() + 1, self.max_visit, self.num_nodes)).to(edge_index.device)
            else:
                attn = torch.ones((batch.max().item() + 1, self.max_visit, self.num_nodes)).to(edge_index.device)

            attn = torch.sum(attn, dim=1)
            xj_node_ids = node_ids[edge_index[0]]
            xj_batch = batch[edge_index[0]]
            attn = attn[xj_batch, xj_node_ids].reshape(-1, 1)

            if self.gnn == "BAT":
                x, w_rel = self.conv[str(layer)](x, edge_index, edge_attr, attn=attn)
            else:
                x = self.conv[str(layer)](x, edge_index)

            # x = self.bn_gnn[str(layer)](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

            if store_attn:
                self.alpha_weights.append(alpha)
                self.beta_weights.append(beta)
                self.attention_weights.append(attn)
                self.edge_weights.append(w_rel)

        if self.patient_mode == "joint" or self.patient_mode == "graph":
            # patient graph embedding through global mean pooling
            x_graph = global_mean_pool(x, batch)  # (B,D)
            x_graph = F.dropout(x_graph, p=self.dropout, training=self.training)

        if self.patient_mode == "joint" or self.patient_mode == "node":
            # patient node embedding through local (direct EHR) mean pooling
            x_node = torch.stack([
                ehr_nodes[i].view(1, -1) @ self.node_emb.weight / torch.sum(ehr_nodes[i])
                for i in range(batch.max().item() + 1)
            ])

            x_node = self.lin(x_node).squeeze(1)
            x_node = F.dropout(x_node, p=self.dropout, training=self.training)  # (B,D)

        if self.patient_mode == "joint":
            # concatenate patient graph embedding and patient node embedding
            x_concat = torch.cat((x_graph, x_node), dim=1)
            x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)

            # MLP for prediction
            logits = self.MLP(x_concat)

        elif self.patient_mode == "graph":
            # MLP for prediction
            logits = self.MLP(x_graph)

        elif self.patient_mode == "node":
            # MLP for prediction
            logits = self.MLP(x_node)

        return {'logits': [logits], 'prototypes': None, 'embeddings': None, 'scores': None}


class MPCare_Pretrain(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.device = args['device']
        self.num_nodes = args['num_nodes']
        self.num_edges = args['num_edges']
        self.num_visits = args['visit_thresh']
        self.num_codes = args['visit_code_num']
        self.out_dim = args['out_dim']
        self.start_embed_dim = args['start_embed_dim']
        self.gnn_type = args['gnn_type']
        self.gnn_layer = args['gnn_layer']
        self.hidden_dim = args['hidden_dim']
        self.head_num = args['head_num']
        self.depth = args['depth']
        self.set_trans_out_num = args['set_trans_out_num']
        self.max_weeks_between = args['max_weeks_between']
        node_embed = args['global_node_attr']
        edge_embed = args['global_edge_attr']
        edge_index = args['global_edge_index']
        edge_ids = args['global_edge_ids']

        self.full_node_ids = torch.arange(self.num_nodes + 1).to(self.device)
        self.full_edge_ids = torch.arange(self.num_edges + 1).to(self.device)

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes + 1, self.start_embed_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)
            self.node_projection = nn.Linear(node_embed.size(-1), self.start_embed_dim)

        if edge_embed is None:
            self.edge_embed = nn.Embedding(self.num_edges + 1, self.start_embed_dim, padding_idx=0)
        else:
            edge_embed = edge_embed.to(torch.float32)
            self.edge_embed = nn.Embedding.from_pretrained(edge_embed, freeze=False, padding_idx=0)
            self.edge_projection = nn.Linear(edge_embed.size(-1), self.start_embed_dim)

        self.edge_index = edge_index.to(self.device)
        self.edge_ids = edge_ids

        self.set_trans_embed = CodeTypeEmbedding(4, self.hidden_dim, padding_idx=0)
        self.gru_embed = DataEmbedding(self.num_visits, self.max_weeks_between, self.hidden_dim)

        if self.gnn_type == 'GAT':
            self.gnn = nn.ModuleList([GATConv(self.hidden_dim, self.hidden_dim) for _ in range(self.gnn_layer)])
        elif self.gnn_type == 'CompGCN':
            self.gnn = nn.ModuleList([CompGCNConv(self.hidden_dim, self.hidden_dim) for _ in range(self.gnn_layer)])

        self.set_transformer = SetTransformer(dim_input=self.hidden_dim,
                                              num_outputs=self.set_trans_out_num,
                                              dim_output=self.hidden_dim,
                                              num_visits=self.num_visits,
                                              dim_hidden=self.hidden_dim,
                                              num_heads=self.head_num,
                                              encoder_depth=self.depth,
                                              decoder_depth=self.depth,
                                              qkv_length=4)

        self.h_0 = nn.Parameter(torch.Tensor(1, 1, self.hidden_dim))
        nn.init.xavier_uniform_(self.h_0)

        self.gru = nn.GRU(input_size=self.hidden_dim,
                          hidden_size=self.hidden_dim,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=False)

        self.visit_projector = nn.Linear(self.hidden_dim, self.out_dim)
        self.patient_projector = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, visit_order, attn_mask, **kwargs):

        x = self.node_embed(self.full_node_ids)  # (node_num,D)
        edge_attr = self.edge_embed(self.full_edge_ids)  # (edge_num,D)

        x = self.node_projection(x)
        edge_attr = self.edge_projection(edge_attr)

        # Global GNN
        if self.gnn_type == 'GAT':
            for layer in self.gnn:
                x = layer(x, self.edge_index, edge_attr=edge_attr)
        elif self.gnn_type == 'CompGCN':
            for layer in self.gnn:
                x, edge_attr = layer(x, self.edge_index, self.edge_ids, edge_attr)
        code_embed = x.clone()  # code_representaiton: (node_num,D)

        B, V, N = node_ids.shape
        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim))

        # Set Transformer
        visit_node_type = kwargs['visit_node_type']
        type_embed = self.set_trans_embed(visit_node_type)
        x, _, _ = self.set_transformer(x, type_embed, attn_mask=attn_mask, return_scores=False)  # (B,V,1,D)
        x = x.squeeze(-2)
        visit_embed = x.clone()

        out_1 = self.visit_projector(x)  # (B,V,node_num)

        # GRU
        data_embed = self.gru_embed(visit_order, visit_times)
        x = x + data_embed
        x, _ = self.gru(x, self.h_0.repeat(1, B, 1))

        visit_mask = attn_mask.all(dim=-1)
        visit_mask = ~visit_mask

        gru_mask = visit_mask.to(int)
        gru_mask = torch.sum(gru_mask, dim=-1) - 1
        x = x[torch.arange(B), gru_mask]
        patient_embed = x.clone()  # Patient Representation: (B,D)

        out_2 = self.patient_projector(x)

        visit_embed_mask = visit_mask.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        visit_embed = visit_embed[visit_embed_mask].view(-1, self.hidden_dim)  # visit representation: (Nv,D)

        set_trans__mask = visit_mask.unsqueeze(-1).expand(-1, -1, self.num_nodes + 1)
        out_1 = out_1[set_trans__mask].view(-1, self.num_nodes + 1)

        embeddings = {'patient_embed': patient_embed, 'visit_embed': visit_embed, 'code_embed': code_embed}

        return {'logits': [out_1, out_2], 'prototypes': None, 'embeddings': embeddings, 'scores': None}


class MPCare_Finetune(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.device = args['device']
        self.num_nodes = args['num_nodes']
        self.num_edges = args['num_edges']
        self.num_visits = args['visit_thresh']
        self.num_codes = args['visit_code_num']
        self.out_dim = args['out_dim']
        self.class_num = args['class_num']
        self.start_embed_dim = args['start_embed_dim']
        self.gnn_type = args['gnn_type']
        self.gnn_layer = args['gnn_layer']
        self.hidden_dim = args['hidden_dim']
        self.head_num = args['head_num']
        self.depth = args['depth']
        self.imp_depth = args['imp_depth']
        self.set_trans_out_num = args['set_trans_out_num']
        self.max_weeks_between = args['max_weeks_between']
        self.multi_level_embed = args['multi_level_embed']
        self.code_proto_num = args['code_proto_num']
        self.visit_proto_num = args['visit_proto_num']
        self.patient_proto_num = args['patient_proto_num']
        self.learnable_proto = args['learnable_proto']
        self.join_method = args['join_method']
        self.residual = args['residual']
        node_embed = args['global_node_attr']
        edge_embed = args['global_edge_attr']
        edge_index = args['global_edge_index']
        edge_ids = args['global_edge_ids']

        self.full_node_ids = torch.arange(self.num_nodes + 1).to(self.device)
        self.full_edge_ids = torch.arange(self.num_edges + 1).to(self.device)
        self.node_projection = None
        self.edge_projection = None

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes + 1, self.start_embed_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)
            self.node_projection = nn.Linear(node_embed.size(-1), self.start_embed_dim)

        if edge_embed is None:
            self.edge_embed = nn.Embedding(self.num_edges + 1, self.start_embed_dim, padding_idx=0)
        else:
            edge_embed = edge_embed.to(torch.float32)
            self.edge_embed = nn.Embedding.from_pretrained(edge_embed, freeze=False, padding_idx=0)
            self.edge_projection = nn.Linear(edge_embed.size(-1), self.start_embed_dim)

        self.edge_index = edge_index.to(self.device)
        self.edge_ids = edge_ids

        if self.multi_level_embed is None:
            self.visit_prototype_labels = None
            self.patient_prototype_labels = None
            self.visit_prototypes = nn.Parameter(torch.Tensor(self.visit_proto_num, self.hidden_dim),
                                                 requires_grad=True)
            self.patient_prototypes = nn.Parameter(torch.Tensor(self.patient_proto_num, self.hidden_dim),
                                                   requires_grad=True)
            nn.init.xavier_uniform_(self.visit_prototypes)
            nn.init.xavier_uniform_(self.patient_prototypes)
        else:
            visit_embed = self.multi_level_embed['visit_embed']
            patient_embed = self.multi_level_embed['patient_embed']
            visit_embed = torch.cat(visit_embed, dim=0)
            patient_embed = torch.cat(patient_embed, dim=0)

            visit_prototypes, self.visit_prototype_labels = self.kmeans(visit_embed, self.visit_proto_num, self.device)
            patient_prototypes, self.patient_prototype_labels = self.kmeans(patient_embed, self.patient_proto_num,
                                                                            self.device)

            visit_prototypes = visit_prototypes.to(torch.float32)
            patient_prototypes = patient_prototypes.to(torch.float32)

            self.visit_prototypes = nn.Parameter(visit_prototypes, requires_grad=self.learnable_proto)
            self.patient_prototypes = nn.Parameter(patient_prototypes, requires_grad=self.learnable_proto)

        self.set_trans_embed = CodeTypeEmbedding(4, self.hidden_dim, padding_idx=0)
        self.gru_embed = DataEmbedding(self.num_visits, self.max_weeks_between, self.hidden_dim)

        if self.gnn_type == 'GAT':
            self.gnn = nn.ModuleList([GATConv(self.hidden_dim, self.hidden_dim) for _ in range(self.gnn_layer)])
        elif self.gnn_type == 'CompGCN':
            self.gnn = nn.ModuleList([CompGCNConv(self.hidden_dim, self.hidden_dim) for _ in range(self.gnn_layer)])

        self.set_transformer = SetTransformer(dim_input=self.hidden_dim,
                                              num_outputs=self.set_trans_out_num,
                                              dim_output=self.hidden_dim,
                                              num_visits=self.num_visits,
                                              dim_hidden=self.hidden_dim,
                                              num_heads=self.head_num,
                                              encoder_depth=self.depth,
                                              decoder_depth=self.depth,
                                              qkv_length=4)

        self.h_0 = nn.Parameter(torch.Tensor(1, 1, self.hidden_dim))
        nn.init.xavier_uniform_(self.h_0)

        self.gru = nn.GRU(input_size=self.hidden_dim,
                          hidden_size=self.hidden_dim,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=False)

        self.code_attn = ImportanceAttention(hidden_dim=self.hidden_dim, num_heads=self.head_num, depth=self.imp_depth)
        self.visit_attn = ImportanceAttention(hidden_dim=self.hidden_dim, num_heads=self.head_num, depth=self.imp_depth)
        self.patient_attn = ImportanceAttention(hidden_dim=self.hidden_dim,
                                                num_heads=self.head_num,
                                                depth=self.imp_depth)

        if self.join_method == 'add':
            self.head = nn.Linear(self.hidden_dim, self.out_dim)
        elif self.join_method == 'concat':
            self.head = nn.Linear(3 * self.hidden_dim, self.out_dim)

    def kmeans(self, X, k, device, max_iters=1000):

        X = X.to(device)
        centroids = X[torch.randint(0, X.size(0), (k,))].to(device)

        for _ in range(max_iters):
            distances = torch.cdist(X, centroids)
            labels = torch.argmin(distances, dim=1)

            new_centroids = torch.stack([
                X[labels == i].mean(dim=0) if
                (labels == i).sum() > 0 else X[torch.randint(0, X.size(0), (1,))].squeeze(0) for i in range(k)
            ])

            if torch.allclose(centroids, new_centroids, atol=1e-4):
                break

            centroids = new_centroids

        return centroids, labels

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, visit_order, attn_mask, **kwargs):

        x = self.node_embed(self.full_node_ids)  # (node_num,D)
        edge_attr = self.edge_embed(self.full_edge_ids)  # (edge_num,D)

        if self.node_projection is not None:
            x = self.node_projection(x)
        if self.edge_projection is not None:
            edge_attr = self.edge_projection(edge_attr)

        # Global GNN
        if self.gnn_type == 'GAT':
            for layer in self.gnn:
                x = layer(x, self.edge_index, edge_attr=edge_attr)
        elif self.gnn_type == 'CompGCN':
            for layer in self.gnn:
                x, edge_attr = layer(x, self.edge_index, self.edge_ids, edge_attr)
        code_embed = x.clone()  # Code Representation: (node_num,D)

        # Get Code Prototypes
        code_prototypes, code_prototype_labels = self.kmeans(x, self.code_proto_num, device=self.device)

        B, V, N = node_ids.shape
        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim))

        # Set Transformer
        visit_node_type = kwargs['visit_node_type']
        type_embed = self.set_trans_embed(visit_node_type)
        x, _, _ = self.set_transformer(x, type_embed, attn_mask=attn_mask, return_scores=False)  # (B,V,1,D)

        x = x.squeeze(-2)
        visit_embed = x.clone()

        # GRU
        data_embed = self.gru_embed(visit_order, visit_times)  # (B,V,D)
        x = x + data_embed
        x, _ = self.gru(x, self.h_0.repeat(1, B, 1))

        visit_mask = attn_mask.all(dim=-1)
        visit_mask = ~visit_mask
        visit_mask = visit_mask.to(int)
        mask = torch.sum(visit_mask, dim=-1) - 1
        x = x[torch.arange(B), mask]
        patient_embed = x.clone()  # Patient Representation: (B,D)

        # Importance Attention
        code_prototypes = F.normalize(code_prototypes, p=2, dim=-1)
        visit_prototypes = F.normalize(self.visit_prototypes, p=2, dim=-1)
        patient_prototypes = F.normalize(self.patient_prototypes, p=2, dim=-1)

        x_code, code_scores = self.code_attn(x, code_prototypes, return_scores=True)
        x_visit, visit_scores = self.visit_attn(x, visit_prototypes, return_scores=True)
        x_patient, patient_scores = self.patient_attn(x, patient_prototypes, return_scores=True)

        if self.join_method == 'add':
            if self.residual:
                x = x + x_code + x_visit + x_patient  # (B,D)
            else:
                x = x_code + x_visit + x_patient

        elif self.join_method == 'concat':
            if self.residual:
                x_code = x_code + x
                x_visit = x_visit + x
                x_patient = x_patient + x
            x = torch.cat((x_code, x_visit, x_patient), dim=-1)  # (B,3D)

        # Prediction
        out = self.head(x)

        visit_mask = attn_mask.all(dim=-1)
        visit_mask = ~visit_mask
        visit_embed_mask = visit_mask.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        visit_embed = visit_embed[visit_embed_mask].view(-1, self.hidden_dim)  # Visit Representation: (Nv,D)

        prototypes = {
            'patient_prototypes': self.patient_prototypes,
            'visit_prototypes': self.visit_prototypes,
            'code_prototypes': code_prototypes,
            'patient_prototype_labels': self.patient_prototype_labels,
            'visit_prototype_labels': self.visit_prototype_labels,
            'code_prototype_labels': code_prototype_labels
        }

        embeddings = {'patient_embed': patient_embed, 'visit_embed': visit_embed, 'code_embed': code_embed}

        scores = {
            'patient_prototype_scores': patient_scores,
            'visit_prototype_scores': visit_scores,
            'code_prototype_scores': code_scores
        }

        return {'logits': [out], 'prototypes': prototypes, 'embeddings': embeddings, 'scores': scores}


MODELS = {
    'MPCare_Pretrain': MPCare_Pretrain,
    'MPCare_Finetune': MPCare_Finetune,
    'GraphCare': GraphCare,
    'GRASP': GRASP,
    'StageNet': StageNet,
    'AdaCare': AdaCare,
    'Deepr': Deepr,
    'Transformer': Transformer,
    'GRU': GRU
}
