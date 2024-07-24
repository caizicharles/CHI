import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GINEConv, GATConv, GINConv, global_mean_pool

from .modules import DataEmbedding, SetTransformer, PrototypeLearner, TimeTransformer, PredictionHead,\
BiAttentionGNNConv, CausalConv1d, Recalibration, DeeprLayer, TemporalEmbedding


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

        self.full_node_ids = torch.arange(self.num_nodes + 1).to(device)
        self.node_embed = nn.Embedding(self.num_nodes + 1, self.input_dim, padding_idx=0)

        self.h_0 = nn.Parameter(torch.randn(self.num_layers, 1, self.hidden_dim)).to(device)
        self.gru = nn.GRU(input_size=self.visit_code_num * self.input_dim,
                          hidden_size=self.hidden_dim,
                          num_layers=self.num_layers,
                          batch_first=True)
        self.head = nn.Linear(self.visit_thresh * self.hidden_dim, self.out_dim)

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, visit_order, attn_mask):

        x = self.node_embed(self.full_node_ids)
        B, V, N = node_ids.shape
        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.input_dim))
        x = x.reshape(B, V, N * self.input_dim)  # (B,V,N*D)

        x, h = self.gru(x, self.h_0.repeat(1, B, 1))  # (B,V,D), (layer_num,B,D)
        x = x.reshape(B, V * self.hidden_dim)

        out = self.head(x)

        return out, None


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
        self.full_node_ids = torch.arange(self.num_nodes + 1).to(device)

        self.node_embed = nn.Embedding(self.num_nodes + 1, self.input_dim, padding_idx=0)
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
        self.head = nn.Linear(self.visit_thresh * self.input_dim, self.out_dim)

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, visit_order, attn_mask):

        B, V, N = node_ids.shape
        x = self.node_embed(self.full_node_ids)
        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.input_dim))
        x = x.reshape(B, V, N * self.input_dim)  # (B,V,N*D)

        visit_mask = attn_mask.all(dim=-1)
        # visit_mask = ~visit_mask
        # mask = visit_mask.unsqueeze(1).unsqueeze(-1).repeat(1, self.head_num, 1, 1)  # (B,head_num,V,1)
        # mask = mask.reshape(B * self.head_num, V, 1).float()
        # mask = mask.matmul(mask.transpose(-1, -2))
        # mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.))

        x = self.encoder(x, src_key_padding_mask=visit_mask)
        x = self.decoder(x, x, tgt_key_padding_mask=visit_mask)

        x = self.fc(x)
        x = x.reshape(B, V * self.input_dim)
        out = self.head(x)

        return out, None


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

        self.full_node_ids = torch.arange(self.num_nodes + 1).to(device)
        self.node_embed = nn.Embedding(self.num_nodes + 1, self.start_embed_dim, padding_idx=0)
        self.gap_time_embed = TemporalEmbedding(self.max_weeks_between, self.start_embed_dim)

        self.layer = DeeprLayer(feature_size=self.start_embed_dim, window=self.window_size, hidden_size=self.hidden_dim)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, visit_order, attn_mask):

        B, V, N = node_ids.shape
        x = self.node_embed(self.full_node_ids)
        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.start_embed_dim))  # (B,V,N,D)

        gap_embed = self.gap_time_embed(visit_times)
        first_row = gap_embed[:, 0, :]
        rest_rows = gap_embed[:, 1:, :]
        gap_embed = torch.cat((rest_rows, first_row.unsqueeze(1)), dim=1)
        gap_embed = gap_embed.unsqueeze(-2)

        visit_mask = attn_mask.all(dim=-1)  # (B,V)
        visit_mask = ~visit_mask

        x = torch.cat((x, gap_embed), dim=-2)
        x = self.layer(x, visit_mask.float())  # (B,D)

        out = self.head(x)

        return out, None


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
        self.dropout = 0.

        self.full_node_ids = torch.arange(self.num_nodes + 1).to(self.device)
        self.node_embed = nn.Embedding(self.num_nodes + 1, self.input_dim, padding_idx=0)

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
        self.nn_output = nn.Linear(self.visit_thresh * self.hidden_dim, self.out_dim)
        self.nn_dropout = nn.Dropout(self.dropout)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, visit_order, attn_mask):

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
        h_reshape = h.contiguous().view(B, V * self.hidden_dim)
        if self.dropout > 0.0:
            h_reshape = self.nn_dropout(h_reshape)
        output = self.nn_output(h_reshape)

        return output, inputse_att


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
        self.dropconnect = 0.
        self.dropout = 0.
        self.dropres = 0.
        self.chunk_size = self.hidden_dim // self.levels
        assert self.hidden_dim % self.levels == 0

        self.full_node_ids = torch.arange(self.num_nodes + 1).to(self.device)

        self.node_embed = nn.Embedding(self.num_nodes + 1, self.input_dim, padding_idx=0)

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
        self.nn_output = nn.Linear(int(self.visit_thresh * self.conv_dim), int(self.output_dim))

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

    def forward(self, node_ids, edge_idx, edge_attr, time, visit_order, attn_mask):

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
            out, c_out, h_out = self.step(input[:, t, :], c_out, h_out, time[:, t])
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
        # rnn_outputs = rnn_outputs.contiguous().view(-1, rnn_outputs.size(-1))
        if self.dropout > 0.0:
            rnn_outputs = self.nn_dropout(rnn_outputs)

        # Prediction
        rnn_outputs = rnn_outputs.reshape(B, V * self.hidden_dim)
        output = self.nn_output(rnn_outputs)
        # output = output.contiguous().view(B, time_step, self.output_dim)
        # output = torch.sigmoid(output)

        return output, torch.stack(distance)


class GraphCare(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.gnn = args['gnn']
        self.embedding_dim = args['start_embed_dim']
        self.decay_rate = args['decay_rate']
        self.patient_mode = args['patient_mode']
        self.use_alpha = True if args['use_alpha'] == 'True' else False
        self.use_beta = True if args['use_beta'] == 'True' else False
        self.edge_attn = True if args['edge_attn'] == 'True' else False
        self.num_nodes = args['num_nodes']
        self.num_edges = args['num_edges']
        self.max_visit = args['max_visit']
        self.hidden_dim = args['hidden_dim']
        self.out_channels = args['out_dim']
        self.drop_rate = 0.
        self.dropout = 0.
        self.attn_init = None
        self.self_attn = 0.

        j = torch.arange(self.max_visit).float()
        self.lambda_j = torch.exp(self.decay_rate * (self.max_visit - j)).unsqueeze(0).reshape(1, self.max_visit,
                                                                                               1).float()

        self.node_emb = nn.Embedding(self.num_nodes, self.embedding_dim)
        self.edge_emb = nn.Embedding(self.num_edges, self.embedding_dim)
        # if node_emb is None:
            # self.node_emb = nn.Embedding(self.num_nodes, self.embedding_dim)
        # else:
        #     self.node_emb = nn.Embedding.from_pretrained(self.node_emb, freeze=freeze)

        # if rel_emb is None:
        #     self.rel_emb = nn.Embedding(self.num_rels, self.embedding_dim)
        # else:
        #     self.rel_emb = nn.Embedding.from_pretrained(self.rel_emb, freeze=freeze)

        self.lin = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)

        self.alpha_attn = nn.ModuleDict()
        self.beta_attn = nn.ModuleDict()
        self.conv = nn.ModuleDict()
        self.bn_gnn = nn.ModuleDict()

        self.leakyrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.tahh = nn.Tanh()

        for layer in range(1, self.gnn_layer + 1):
            if self.use_alpha:
                self.alpha_attn[str(layer)] = nn.Linear(self.num_nodes, self.num_nodes)

                if attn_init is not None:
                    attn_init = attn_init.float()  # Convert attn_init to float
                    attn_init_matrix = torch.eye(
                        self.num_nodes).float() * attn_init  # Multiply the identity matrix by attn_init
                    self.alpha_attn[str(layer)].weight.data.copy_(
                        attn_init_matrix)  # Copy the modified attn_init_matrix to the weights

                else:
                    nn.init.xavier_normal_(self.alpha_attn[str(layer)].weight)
            if self.use_beta:
                self.beta_attn[str(layer)] = nn.Linear(self.num_nodes, 1)
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
                edge_ids,
                edge_index,
                edge_attr,
                visit_times,
                visit_order,
                visit_node,
                ehr_nodes,
                batch,
                attn_mask,
                store_attn=False,
                in_drop=False):

        # node_ids, rel_ids, edge_index, batch, visit_node, ehr_nodes, store_attn=False, in_drop=False
        # node_ids, edge_idx, edge_attr, visit_times, visit_order, attn_mask
        ## node_ids, edge_ids, edge_index, edge_attr, visit_times, visit_order, visit_node, ehr_nodes, batch, attn_mask, store_attn=False, in_drop=False

        # batch: (node_num,) assigns node to batch
        # visit_node: (B,V,node_num) multihot tensor indicating nodes in all visits
        # ehr_nodes: (B,) multihot tensor indicating whether code belongs to patient

        if in_drop and self.drop_rate > 0:
            edge_count = edge_index.size(1)
            edges_to_remove = int(edge_count * self.drop_rate)
            indices_to_remove = set(random.sample(range(edge_count), edges_to_remove))
            edge_index = edge_index[:,
                                    [i for i in range(edge_count) if i not in indices_to_remove]].to(edge_index.device)
            edge_ids = torch.tensor([rel_id for i, rel_id in enumerate(edge_ids) if i not in indices_to_remove],
                                   device=edge_ids.device)

        x = self.node_emb(node_ids).float()             # (B,V,node_num,D)
        edge_attr = self.edge_emb(edge_ids).float()     # (B,V,edge_num,D)

        x = self.lin(x)
        edge_attr = self.lin(edge_attr)

        if store_attn:
            self.alpha_weights = []
            self.beta_weights = []
            self.attention_weights = []
            self.edge_weights = []

        for layer in range(1, self.layers + 1):
            if self.use_alpha:
                # alpha = masked_softmax((self.leakyrelu(self.alpha_attn[str(layer)](visit_node.float()))), mask=visit_node>1, dim=1)
                alpha = torch.softmax((self.alpha_attn[str(layer)](visit_node.float())),
                                      dim=1)  # (batch, max_visit, num_nodes)

            if self.use_beta:
                # beta = masked_softmax((self.leakyrelu(self.beta_attn[str(layer)](visit_node.float()))), mask=visit_node>1, dim=0) * self.lambda_j
                beta = torch.tanh((self.beta_attn[str(layer)](visit_node.float()))) * self.lambda_j

            if self.use_alpha and self.use_beta:
                attn = alpha * beta
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
                x, w_rel = self.conv[str(layer)](x, edge_index, edge_attr, attn=attn)   # (,D)

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
            x_graph = global_mean_pool(x, batch)
            x_graph = F.dropout(x_graph, p=self.dropout, training=self.training)

        if self.patient_mode == "joint" or self.patient_mode == "node":
            # patient node embedding through local (direct EHR) mean pooling
            x_node = torch.stack([
                ehr_nodes[i].view(1, -1) @ self.node_emb.weight / torch.sum(ehr_nodes[i])
                for i in range(batch.max().item() + 1)
            ])
            x_node = self.lin(x_node).squeeze(1)
            x_node = F.dropout(x_node, p=self.dropout, training=self.training)

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

        if store_attn:
            return logits, self.alpha_weights, self.beta_weights, self.attention_weights, self.edge_weights
        else:
            return logits


class OurModel(nn.Module):

    def __init__(self, args):

        super().__init__()

        device = args['device']
        self.num_nodes = args['num_nodes']
        self.num_edges = args['num_nodes']
        num_visits = args['visit_thresh']
        max_weeks_between = args['max_weeks_between']
        start_embed_dim = args['start_embed_dim']
        self.gnn_hidden_dim = args['gnn_hidden_dim']
        set_trans_hidden_dim = args['set_trans_hidden_dim']
        set_trans_out_dim = args['set_trans_out_dim']
        proto_hidden_dim = args['proto_hidden_dim']
        proto_out_dim = args['proto_out_dim']
        time_trans_hidden_dim = args['time_trans_hidden_dim']
        out_dim = args['out_dim']
        self.gnn_layer = args['gnn_layer']
        proto_num = args['proto_num']
        set_head_num = args['set_head_num']
        set_num_inds = args['visit_thresh']
        set_encoder_depth = args['set_encoder_depth']
        set_decoder_depth = args['set_decoder_depth']
        set_trans_out_num = args['set_trans_out_num']
        proto_head_num = args['proto_head_num']
        proto_depth = args['proto_depth']
        time_head_num = args['time_head_num']
        time_depth = args['time_depth']
        node_embed = None

        self.full_node_ids = torch.arange(self.num_nodes + 1).to(device)

        self.prototypes = nn.Parameter(torch.randn(proto_num, set_trans_out_dim))

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes + 1, start_embed_dim, padding_idx=0)
        '''
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=freeze)

        if edge_embed is None:
            self.edge_embed = nn.Embedding(num_edges+1, embed_dim)
        else:
            self.edge_embed = nn.Embedding.from_pretrained(edge_embed, freeze=freeze)
        '''

        self.time_trans_embed = DataEmbedding(num_visits, max_weeks_between, proto_out_dim)

        self.fc = nn.Linear(start_embed_dim, self.gnn_hidden_dim)

        self.gnn = nn.ModuleList([GATConv(self.gnn_hidden_dim, self.gnn_hidden_dim) for _ in range(self.gnn_layer)])

        self.set_transformer = SetTransformer(dim_input=self.gnn_hidden_dim,
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
        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
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


MODELS = {
    'OurModel': OurModel,
    'GraphCare': GraphCare,
    'StageNet': StageNet,
    # 'GRAM': GRAM,
    # 'RETAIN': RETAIN,
    'AdaCare': AdaCare,
    'Deepr': Deepr,
    'Transformer': Transformer,
    'GRU': GRU
}
