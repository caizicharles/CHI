import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GATConv, GINConv, global_mean_pool

from .modules_1 import CompGCNConv, PrototypeLearning, PrototypePrediction,\
    BiAttentionGNNConv, \
    TimeGapEmbedding, \
    TransformerTime, Aggregator

from pyhealth.models import GRASPLayer, StageNetLayer, AdaCareLayer, DeeprLayer, SafeDrugLayer, MICRONLayer


class GRU(nn.Module):

    def __init__(self, args):
        super().__init__()

        device = args['device']
        self.num_nodes = args['num_nodes']
        self.visit_thresh = args['visit_thresh']
        self.visit_code_num = args['visit_code_num']
        self.input_dim = args['input_dim']
        self.hidden_dim = args['hidden_dim']
        self.out_dim = args['out_dim']
        node_embed = args['global_node_attr']

        self.full_node_ids = torch.arange(self.num_nodes + 1).to(device)

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes + 1, self.input_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)
            self.node_projection = nn.Linear(node_embed.size(-1), self.input_dim)

        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, attn_mask, **kwargs):

        x = self.node_embed(self.full_node_ids)
        x = self.node_projection(x)

        B, V, N = node_ids.shape
        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.input_dim))

        inv_attn_mask = ~attn_mask
        mean_mask = inv_attn_mask.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)).to(int)
        x = x * mean_mask
        x = torch.sum(x, dim=2)
        mean_count = inv_attn_mask.to(int).sum(dim=-1, keepdim=True)
        mean_count[mean_count == 0] = 1
        x = x / mean_count  # (B,V,D)

        x, _ = self.gru(x)  # (B,V,D)

        visit_mask = attn_mask.all(dim=-1)
        visit_mask = ~visit_mask
        visit_mask = visit_mask.to(int)
        mask = torch.sum(visit_mask, dim=-1) - 1
        x = x[torch.arange(B), mask]  # (B,D)

        out = self.head(x)

        return {'logits': [out], 'prototypes': None, 'embeddings': None}


class Transformer(nn.Module):

    def __init__(self, args):
        super().__init__()

        device = args['device']
        self.num_nodes = args['num_nodes']
        self.visit_thresh = args['visit_thresh']
        self.visit_code_num = args['visit_code_num']
        self.input_dim = args['input_dim']
        self.ff_dim = args['ff_dim']
        self.head_num = args['head_num']
        self.encoder_depth = args['encoder_depth']
        self.out_dim = args['out_dim']
        node_embed = args['global_node_attr']

        self.full_node_ids = torch.arange(self.num_nodes + 1).to(device)

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes + 1, self.input_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)
            self.node_projection = nn.Linear(node_embed.size(-1), self.input_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.input_dim,
                                       nhead=self.head_num,
                                       dim_feedforward=self.ff_dim,
                                       dropout=0.1,
                                       batch_first=True), self.encoder_depth)

        self.head = nn.Linear(self.input_dim, self.out_dim)

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, attn_mask, **kwargs):

        B, V, N = node_ids.shape
        x = self.node_embed(self.full_node_ids)
        x = self.node_projection(x)

        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.input_dim))

        inv_attn_mask = ~attn_mask
        mean_mask = inv_attn_mask.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)).to(int)
        x = x * mean_mask
        x = torch.sum(x, dim=2)
        mean_count = inv_attn_mask.to(int).sum(dim=-1, keepdim=True)
        mean_count[mean_count == 0] = 1
        x = x / mean_count  # (B,V,D)

        visit_mask = attn_mask.all(dim=-1)
        x = self.encoder(x, src_key_padding_mask=visit_mask)  # (B,V,D)

        inv_visit_mask = ~visit_mask
        mask = inv_visit_mask.to(int).unsqueeze(-1)
        x = x * mask
        x = x.sum(dim=1)
        count = mask.sum(dim=1)
        count[count == 0] = 1
        x = x / count  # (B,D)

        out = self.head(x)

        return {'logits': [out], 'prototypes': None, 'embeddings': None}


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
            self.node_projection = nn.Linear(node_embed.size(-1), self.start_embed_dim)

        self.time_gap_embed = TimeGapEmbedding(self.start_embed_dim, device)

        self.layer = DeeprLayer(feature_size=self.start_embed_dim, window=self.window_size, hidden_size=self.hidden_dim)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, attn_mask, **kwargs):

        B, V, N = node_ids.shape
        x = self.node_embed(self.full_node_ids)
        x = self.node_projection(x)

        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.start_embed_dim))  # (B,V,N,D)

        gap_embed = self.time_gap_embed(visit_times)
        first_row = gap_embed[:, 0]
        rest_rows = gap_embed[:, 1:]
        gap_embed = torch.cat((rest_rows, first_row.unsqueeze(1)), dim=1)
        gap_embed = gap_embed.unsqueeze(-2)
        x = torch.cat((x, gap_embed), dim=-2)  # (B,V,N+1,D)
        x = x.reshape(B, V * (N + 1), x.size(-1))  # (B,V(N+1),D)

        visit_mask = attn_mask.all(dim=-1)
        inv_visit_mask = ~visit_mask
        mask_pad = inv_visit_mask.to(int).unsqueeze(-1)

        inv_attn_mask = ~attn_mask
        mask = inv_attn_mask.to(int)
        mask = torch.cat((mask, mask_pad), dim=-1)
        mask = mask.reshape(B, V * (N + 1))  # (B,V(N+1))

        x = self.layer(x, mask)  # (B,VN,D)

        output = self.head(x)

        return {'logits': [output], 'prototypes': None, 'embeddings': None}


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
            self.node_embed = nn.Embedding(self.num_nodes + 1, self.input_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)
            self.node_projection = nn.Linear(node_embed.size(-1), self.input_dim)

        self.adacare = AdaCareLayer(input_dim=self.input_dim,
                                    hidden_dim=self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    kernel_num=self.kernel_num,
                                    r_v=self.r_v,
                                    r_c=self.r_c)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, attn_mask, **kwargs):

        B, V, N = node_ids.shape
        x = self.node_embed(self.full_node_ids)
        x = self.node_projection(x)

        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.input_dim))

        inv_attn_mask = ~attn_mask
        mean_mask = inv_attn_mask.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)).to(int)
        x = x * mean_mask
        x = torch.sum(x, dim=2)
        mean_count = inv_attn_mask.to(int).sum(dim=-1, keepdim=True)
        mean_count[mean_count == 0] = 1
        x = x / mean_count  # (B,V,D)

        visit_mask = attn_mask.all(dim=-1)
        inv_visit_mask = ~visit_mask
        mask = inv_visit_mask.to(int)

        x, _, _, _ = self.adacare(x, mask)

        output = self.head(x)

        return {'logits': [output], 'prototypes': None, 'embeddings': None}


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
            self.node_embed = nn.Embedding(self.num_nodes + 1, self.input_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)
            self.node_projection = nn.Linear(node_embed.size(-1), self.input_dim)

        self.stagenet = StageNetLayer(input_dim=self.input_dim,
                                      chunk_size=self.hidden_dim,
                                      conv_size=self.conv_size,
                                      levels=self.levels)

        self.head = nn.Linear(self.hidden_dim * self.levels, self.output_dim)

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, attn_mask, **kwargs):

        B, V, N = node_ids.shape
        x = self.node_embed(self.full_node_ids)
        x = self.node_projection(x)

        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.input_dim))

        inv_attn_mask = ~attn_mask
        mean_mask = inv_attn_mask.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)).to(int)
        x = x * mean_mask
        x = torch.sum(x, dim=2)
        mean_count = inv_attn_mask.to(int).sum(dim=-1, keepdim=True)
        mean_count[mean_count == 0] = 1
        x = x / mean_count  # (B,V,D)

        visit_mask = attn_mask.all(dim=-1)
        inv_visit_mask = ~visit_mask
        mask = inv_visit_mask.to(int)

        x, _, _ = self.stagenet(x, time=visit_times, mask=mask)

        output = self.head(x)

        return {'logits': [output], 'prototypes': None, 'embeddings': None}


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

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, attn_mask, **kwargs):

        x = self.node_embed(self.full_node_ids)  # (node_num,D)
        x = self.node_projection(x)

        B, V, N = node_ids.shape
        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim))

        inv_attn_mask = ~attn_mask
        mean_mask = inv_attn_mask.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)).to(int)
        x = x * mean_mask
        x = torch.sum(x, dim=2)
        mean_count = inv_attn_mask.to(int).sum(dim=-1, keepdim=True)
        mean_count[mean_count == 0] = 1
        x = x / mean_count  # (B,V,D)

        visit_mask = attn_mask.all(dim=-1)
        inv_visit_mask = ~visit_mask
        inv_visit_mask = inv_visit_mask.to(int)

        x = self.grasp(x, mask=inv_visit_mask)  # (B,D)

        out = self.head(x)

        return {'logits': [out], 'prototypes': None, 'embeddings': None}


class MICRON(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.device = args['device']
        self.num_nodes = args['num_nodes']
        self.num_edges = args['num_edges']
        self.num_visits = args['visit_thresh']
        self.num_codes = args['visit_code_num']
        self.out_dim = args['out_dim']
        self.input_dim = args['input_dim']
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

        # self.micron = MICRONLayer(input_size=self.input_dim,
        #                           hidden_size=self.hidden_dim,
        #                           num_drugs=self.label_tokenizer.get_vocabulary_size())

        self.head = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, attn_mask, **kwargs):

        x = self.node_embed(self.full_node_ids)  # (node_num,D)
        x = self.node_projection(x)

        B, V, N = node_ids.shape
        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim))

        inv_attn_mask = ~attn_mask
        mean_mask = inv_attn_mask.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)).to(int)
        x = x * mean_mask
        x = torch.sum(x, dim=2)
        mean_count = inv_attn_mask.to(int).sum(dim=-1, keepdim=True)
        mean_count[mean_count == 0] = 1
        x = x / mean_count  # (B,V,D)

        visit_mask = attn_mask.all(dim=-1)
        inv_visit_mask = ~visit_mask
        inv_visit_mask = inv_visit_mask.to(int)

        x = self.grasp(x, mask=inv_visit_mask)  # (B,D)

        out = self.head(x)

        return {'logits': [out], 'prototypes': None, 'embeddings': None}


class KerPrint(nn.Module):

    def __init__(self, args):

        super(KerPrint, self).__init__()

        self.device = args['device']
        self.num_nodes = args['num_nodes']
        self.num_edges = args['num_edges']
        self.num_visits = args['visit_thresh']
        self.num_codes = args['visit_code_num']
        self.out_dim = args['out_dim']
        self.start_embed_dim = args['start_embed_dim']
        self.hidden_dim = args['hidden_dim']
        self.ff_dim = args['ff_dim']
        self.candidate_knowledge_num = args['candidate_knowledge_num']
        node_embed = args['global_node_attr']
        edge_embed = args['global_edge_attr']
        edge_index = args['global_edge_index']

        self.move_num = 1
        self.hita_encoder_head_num = 4
        self.hita_encoder_layers = 1
        self.n_layers_KGAT = 1
        self.kgat_mess_dropout = 0.5
        self.train_dropout_rate = 0.5

        self.full_node_ids = torch.arange(self.num_nodes + 1).to(self.device)
        self.full_edge_ids = torch.arange(self.num_edges + 1).to(self.device)
        self.edge_index = edge_index.to(self.device)

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

        self.hita_time_selection_layer_global = [self.hidden_dim, self.hidden_dim]
        self.hita_time_selection_layer_encoder = [self.hidden_dim, self.hidden_dim]
        self.transformerEncoder = TransformerTime(
            hita_input_size=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            ff_dim=self.ff_dim,
            max_seq_len=self.num_visits,
            move_num=self.move_num,
            hita_time_selection_layer_global=self.hita_time_selection_layer_global,
            hita_time_selection_layer_encoder=self.hita_time_selection_layer_encoder,
            hita_encoder_head_num=self.hita_encoder_head_num,
            hita_encoder_layers=self.hita_encoder_layers)

        self.element_weight_matrix = nn.Parameter(
            torch.Tensor(self.candidate_knowledge_num, self.hidden_dim, self.hidden_dim))
        self.patient_transform_fc = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.aggregator_layers_KGAT = nn.ModuleList()
        for _ in range(self.n_layers_KGAT):
            self.aggregator_layers_KGAT.append(Aggregator(self.hidden_dim, self.hidden_dim, self.kgat_mess_dropout))

        self.classfier_fc = nn.Linear(2 * self.hidden_dim, self.out_dim)

    def calc_knowledge_embed(self, sequence_embedding_final, canknowledge_embed):
        candidate_knowledge_num = canknowledge_embed.size(1)
        patient_element_matrix = sequence_embedding_final.unsqueeze(2).expand(-1, -1, sequence_embedding_final.size(1))
        patient_element_matrix_expand = patient_element_matrix.unsqueeze(1).expand(-1, candidate_knowledge_num, -1, -1)
        knowledge_element_matrix = canknowledge_embed.unsqueeze(2).expand(-1, -1, sequence_embedding_final.size(1), -1)
        interaction_matrix = patient_element_matrix_expand * knowledge_element_matrix + patient_element_matrix_expand - knowledge_element_matrix
        element_attention_matrix = torch.softmax(torch.sum(interaction_matrix * self.element_weight_matrix, dim=2),
                                                 dim=1)
        result_embedding = torch.sum(canknowledge_embed * element_attention_matrix, dim=1)
        return result_embedding, element_attention_matrix

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, attn_mask, **kwargs):

        x = self.node_embed(self.full_node_ids)  # (node_num,D)
        edge_attr = self.edge_embed(self.full_edge_ids)  # (edge_num,D)

        if self.node_projection is not None:
            x = self.node_projection(x)
        if self.edge_projection is not None:
            edge_attr = self.edge_projection(edge_attr)

        node_embed = x.clone()

        B, V, N = node_ids.shape
        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim))

        inv_attn_mask = ~attn_mask
        mean_mask = inv_attn_mask.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)).to(int)
        x = x * mean_mask
        x = torch.sum(x, dim=2)
        mean_count = inv_attn_mask.to(int).sum(dim=-1, keepdim=True)
        mean_count[mean_count == 0] = 1
        x = x / mean_count  # (B,V,D)

        visit_mask = attn_mask.all(dim=-1)
        inv_visit_mask = ~visit_mask
        mask = inv_visit_mask.to(int).unsqueeze(-1)
        visit_times = visit_times.to(torch.float32).unsqueeze(-1)

        sequence_embedding_final, _ = self.transformerEncoder(x, visit_times, mask, 1, self.device)
        exit()
        # kgat_embed = []

        # for i, layer in enumerate(self.aggregator_layers_KGAT):
        #     global_node_embed = layer(node_embed, edge_index, edge_attr)
        #     norm_embed = F.normalize(global_node_embed, p=2, dim=1)
        #     kgat_embed = [norm_embed]
        # kgat_embed = torch.cat(kgat_embed, dim=0)

        # canknowledge_index = canknowledge_batch.unsqueeze(2).expand(-1, -1, kgat_embed.size(1))
        # kgat_embed = kgat_embed.expand(canknowledge_index.size(0), kgat_embed.size(0), kgat_embed.size(1))
        # canknowledge_embed = torch.gather(kgat_embed, 1, canknowledge_index)
        # sequence_embedding_transform = self.relu(self.patient_transform_fc(sequence_embedding_final))
        # knowledge_embed, _ = self.calc_knowledge_embed(sequence_embedding_transform, canknowledge_embed)

        # sequence_embedding_all = torch.cat((sequence_embedding_final, knowledge_embed), dim=1)
        # sequence_embedding_all = self.dropout(sequence_embedding_all)

        return  # sequence_embedding_all

    # def calc_sequence_logits(self, mode, g_global, g_personal, x_batch, s_batch, s_batch_dim2, miss_pair, batch_mask,
    #                          batch_mask_final, seq_time_batch, canknowledge_batch):

    #     all_sequence_embed = self.calc_sequence_embedding(mode, g_global, g_personal, x_batch, s_batch, s_batch_dim2,
    #                                                       batch_mask, batch_mask_final, seq_time_batch,
    #                                                       canknowledge_batch)

    #     all_sequence_logits = self.classfier_fc(all_sequence_embed)

    #     return all_sequence_logits

    # def forward(self, mode, *input):
    #     return self.calc_sequence_logits(mode, *input)


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
        self.dropout = args['dropout']
        self.attn_init = args['attn_init']
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

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, attn_mask, in_drop=False, store_attn=False, **kwargs):

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

        return {'logits': [logits], 'prototypes': None, 'embeddings': None}


class ProtoEHR(nn.Module):
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
        self.ff_dim = args['ff_dim']
        self.head_num = args['head_num']
        self.depth = args['depth']
        self.dropout = args['dropout']
        self.max_weeks_between = args['max_weeks_between']
        node_embed = args['global_node_attr']
        edge_embed = None
        edge_index = args['global_edge_index']
        edge_ids = args['global_edge_ids']
        edge_type = args['global_edge_type']
        edge_norm = args['global_edge_norm']
        graph = args['graph']

        unique_edge_num = len(torch.unique(edge_type))
        self.full_node_ids = torch.arange(self.num_nodes + 1).to(self.device)
        self.full_edge_ids = torch.arange(unique_edge_num).to(self.device)
        
        self.edge_type = edge_type.to(self.device)
        self.edge_norm = edge_norm.to(self.device)
        self.graph = graph.to(self.device)
        self.node_projection = None
        self.edge_projection = None

        self.patient_proto_num = args['patient_proto_num']
        self.visit_proto_num = args['visit_proto_num']
        self.code_proto_num = args['code_proto_num']
        self.prototype_learning_mode = args['prototype_learning_mode']
        self.prototype_prediction_mode = args['prototype_prediction_mode']
        self.softmax_temp = args['softmax_temp']
        self.commitment_cost = None
        self.attn_iter = None

        if self.prototype_learning_mode == 'VQVAE' or self.prototype_learning_mode == 'onehot':
            self.commitment_cost = args['commitment_cost']
        elif self.prototype_learning_mode == 'QKV':
            self.attn_iter = args['attn_iter']

        if node_embed is None:
            self.node_embed = nn.Embedding(self.num_nodes + 1, self.start_embed_dim, padding_idx=0)
        else:
            self.node_embed = nn.Embedding.from_pretrained(node_embed, freeze=False, padding_idx=0)
            self.node_projection = nn.Linear(node_embed.size(-1), self.start_embed_dim)

        if edge_embed is None:
            self.edge_embed = nn.Embedding(unique_edge_num, self.start_embed_dim, padding_idx=0)
        else:
            edge_embed = edge_embed.to(torch.float32)
            self.edge_embed = nn.Embedding.from_pretrained(edge_embed, freeze=False, padding_idx=0)
            self.edge_projection = nn.Linear(edge_embed.size(-1), self.start_embed_dim)

        self.edge_index = edge_index.to(self.device)
        self.edge_ids = edge_ids

        if self.gnn_type == 'GAT':
            self.gnn = nn.ModuleList([GATConv(self.hidden_dim, self.hidden_dim) for _ in range(self.gnn_layer)])
        elif self.gnn_type == 'CompGCN':
            self.gnn = nn.ModuleList([
                CompGCNConv(self.hidden_dim, self.hidden_dim, torch.tanh, drop_rate=self.dropout, opn='rotate')
                for _ in range(self.gnn_layer)
            ])

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hidden_dim,
                                       nhead=self.head_num,
                                       dim_feedforward=self.ff_dim,
                                       dropout=self.dropout,
                                       batch_first=True), self.depth)

        self.patient_PtL = PrototypeLearning(prototype_num=self.patient_proto_num,
                                          hidden_dim=self.hidden_dim,
                                          commitment_cost=self.commitment_cost,
                                          attn_iter=self.attn_iter,
                                          learning_mode=self.prototype_learning_mode,
                                          prediction_mode=self.prototype_prediction_mode)
        
        self.visit_PtL = PrototypeLearning(prototype_num=self.visit_proto_num,
                                          hidden_dim=self.hidden_dim,
                                          commitment_cost=self.commitment_cost,
                                          attn_iter=self.attn_iter,
                                          learning_mode=self.prototype_learning_mode,
                                          prediction_mode=self.prototype_prediction_mode)
        
        self.code_PtL = PrototypeLearning(prototype_num=self.code_proto_num,
                                          hidden_dim=self.hidden_dim,
                                          commitment_cost=self.commitment_cost,
                                          attn_iter=self.attn_iter,
                                          learning_mode=self.prototype_learning_mode,
                                          prediction_mode=self.prototype_prediction_mode)

        self.patient_PtP = PrototypePrediction(prototype_num=self.patient_proto_num,
                                               hidden_dim=self.hidden_dim,
                                               out_dim=self.out_dim,
                                               prediction_mode=self.prototype_prediction_mode,
                                               learning_mode=self.prototype_learning_mode)
        
        self.visit_PtP = PrototypePrediction(prototype_num=self.visit_proto_num,
                                               hidden_dim=self.hidden_dim,
                                               out_dim=self.out_dim,
                                               prediction_mode=self.prototype_prediction_mode,
                                               learning_mode=self.prototype_learning_mode)
        
        self.code_PtP = PrototypePrediction(prototype_num=self.code_proto_num,
                                               hidden_dim=self.hidden_dim,
                                               out_dim=self.out_dim,
                                               prediction_mode=self.prototype_prediction_mode,
                                               learning_mode=self.prototype_learning_mode)
        
        self.code_PtF = nn.MultiheadAttention(embed_dim=self.hidden_dim,
                                              num_heads=1,
                                              dropout=self.dropout,
                                              batch_first=True)
        
        self.visit_PtF = nn.MultiheadAttention(embed_dim=self.hidden_dim,
                                              num_heads=1,
                                              dropout=self.dropout,
                                              batch_first=True)
        
        self.patient_PtF = nn.MultiheadAttention(embed_dim=self.hidden_dim,
                                              num_heads=1,
                                              dropout=self.dropout,
                                              batch_first=True)
        
        self.PtF_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.PtF_weights = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
        nn.init.xavier_normal_(self.PtF_weights)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)
        
    def prototype_fusion(self, embedding, code_prototypes, visit_prototypes, patient_prototypes):
        x_code, _ = self.code_PtF(embedding, code_prototypes, code_prototypes)
        x_visit, _ = self.visit_PtF(embedding, visit_prototypes, visit_prototypes)
        x_patient, _ = self.patient_PtF(embedding, patient_prototypes, patient_prototypes)

        code_weight = torch.matmul(x_code, self.PtF_weights)
        visit_weight = torch.matmul(x_visit, self.PtF_weights)
        patient_weight = torch.matmul(x_patient, self.PtF_weights)
        weights = torch.concat((code_weight, visit_weight, patient_weight), dim=-1)
        weights = torch.softmax(weights/self.softmax_temp, dim=-1).unsqueeze(dim=1) # (B,3,1)

        x_code = x_code.unsqueeze(dim=1)  # (B,1,D)
        x_visit = x_visit.unsqueeze(dim=1)  # (B,1,D)
        x_patient = x_patient.unsqueeze(dim=1)  # (B,1,D)
        x = torch.concat((x_code, x_visit, x_patient), dim=1)   # (B,3,D)
        x = torch.matmul(weights, x).squeeze(dim=1)

        return x, weights.squeeze(dim=1)

    def forward(self, node_ids, edge_idx, edge_attr, visit_times, attn_mask, **kwargs):
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
                x, edge_attr = layer(self.graph, x, edge_attr, self.edge_type, self.edge_norm)

        _, _, code_prototypes = self.code_PtL(x)

        # Select
        B, V, N = node_ids.shape
        x = x.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)  # (B,V,node_num,D)
        x = torch.gather(x, 2, node_ids.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim))  # (B,V,N,D)
        
        x = x.reshape(-1, x.size(-1))
        x_code = self.code_PtP(x, code_prototypes)
        x_code = x_code.reshape(B, V, N, x_code.size(-1))
        x = x.reshape(B, V, N, x.size(-1))
        x = x + x_code

        # AvgPool
        inv_attn_mask = ~attn_mask
        mean_mask = inv_attn_mask.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)).to(int)
        x = x * mean_mask
        x = torch.sum(x, dim=2)
        mean_count = inv_attn_mask.to(int).sum(dim=-1, keepdim=True)
        mean_count[mean_count == 0] = 1
        x = x / mean_count  # (B,V,D)

        visit_mask = attn_mask.all(dim=-1)
        visit_proto_mask = visit_mask.flatten()
        visit_proto_x = x.reshape(-1, x.size(-1))
        visit_proto_x = visit_proto_x[~visit_proto_mask]

        _, _, visit_prototypes = self.visit_PtL(visit_proto_x)
        x = x.reshape(-1, x.size(-1))
        x_visit = self.visit_PtP(x, visit_prototypes)
        x_visit = x_visit.reshape(B, V, x_visit.size(-1))
        x = x.reshape(B, V, x.size(-1))
        x = x + x_visit

        # Transformer
        transformer_mask = attn_mask.all(dim=-1)
        x = self.transformer_encoder(x, src_key_padding_mask=transformer_mask)

        # Retrive Last Visit
        visit_mask = attn_mask.all(dim=-1)
        inv_visit_mask = ~visit_mask
        last_visit_mask = inv_visit_mask.to(int)
        last_visit_mask = torch.sum(last_visit_mask, dim=-1) - 1
        x = x[torch.arange(B), last_visit_mask]

        _, _, patient_prototypes = self.patient_PtL(x)
        x_patient = self.patient_PtP(x, patient_prototypes)
        x = x + x_patient
        pre_PtF_embeddings = x.clone()

        # Prototype Fusion
        x_fused, weights = self.prototype_fusion(x, self.code_PtL.prototypes, self.visit_PtL.prototypes, self.patient_PtL.prototypes)
        x = x + x_fused
        post_PtF_embeddings = x.clone()
        out = self.head(x)

        prototypes = {'code_prototypes': self.code_PtL.prototypes, 'visit_prototypes': self.visit_PtL.prototypes, 'patient_prototypes': self.patient_PtL.prototypes}
        embeddings = {'pre_PtF_embeddings': pre_PtF_embeddings, 'post_PtF_embeddings': post_PtF_embeddings}

        return {'logits': [out], 'prototypes': prototypes, 'embeddings': embeddings, 'hierarchy_weights': weights}


MODELS = {
    'ProtoEHR': ProtoEHR,
    'GraphCare': GraphCare,
    # 'KerPrint': KerPrint,
    # 'SeqCare': SeqCare(),
    'GRASP': GRASP,
    'StageNet': StageNet,
    'AdaCare': AdaCare,
    'Deepr': Deepr,
    'Transformer': Transformer,
    'GRU': GRU
}
