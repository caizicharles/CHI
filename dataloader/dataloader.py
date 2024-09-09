import torch
from torch_geometric.data import Data, Batch
import random


class DataLoader():

    def __init__(self,
                 dataset: dict,
                 graph: Data,
                 model_name: str,
                 task: str,
                 batch_size: int = 1,
                 shuffle: bool = False):

        self.dataset = dataset.dataset
        self.graph = graph
        self.task = task
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_index = 0

        if model_name == 'GraphCare':
            self.get_additional = True
        else:
            self.get_additional = False

        if self.shuffle:
            self._shuffle_data()

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __iter__(self):
        return self

    def _shuffle_data(self):
        random.shuffle(self.dataset)

    def reindex_edge_index(self, edge_index):
        # Get unique nodes
        unique_nodes = torch.unique(edge_index)

        # Create a mapping from original node indices to new consecutive indices
        node_mapping = {node.item(): i for i, node in enumerate(unique_nodes)}

        # Apply mapping to edge_index
        reindexed_edge_index = edge_index.clone()
        for i in range(reindexed_edge_index.size(1)):
            reindexed_edge_index[0, i] = node_mapping[reindexed_edge_index[0, i].item()]
            reindexed_edge_index[1, i] = node_mapping[reindexed_edge_index[1, i].item()]

        return reindexed_edge_index

    def __next__(self):

        if self.current_index >= len(self.dataset):
            self.current_index = 0
            if self.shuffle:
                self._shuffle_data()
            raise StopIteration

        if self.current_index + self.batch_size > len(self.dataset):
            self.current_index = 0
            if self.shuffle:
                self._shuffle_data()
            raise StopIteration

        grouped_patients = self.dataset[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size

        batch_data = Data()

        batch_data.global_edge_index = self.graph.edge_index
        batch_data.global_edge_attr = self.graph.edge_attr
        batch_data.global_edge_ids = self.graph.edge_ids

        visit_node_ids = [data['visit_node_ids'] for data in grouped_patients]
        visit_rel_times = [data['visit_rel_times'] for data in grouped_patients]
        visit_order = [data['visit_order'] for data in grouped_patients]
        visit_node_type = [data['visit_node_type'] for data in grouped_patients]
        attn_mask = [data['node_padding_mask'] for data in grouped_patients]

        batch_data.visit_node_ids = torch.stack(visit_node_ids)
        batch_data.visit_rel_times = torch.stack(visit_rel_times)
        batch_data.visit_order = torch.stack(visit_order)
        batch_data.visit_node_type = torch.stack(visit_node_type)
        batch_data.attn_mask = torch.stack(attn_mask)

        if self.task != 'pretrain':
            visit_nodes = [data['visit_nodes'].unsqueeze(0) for data in grouped_patients]
            ehr_nodes = [data['ehr_nodes'] for data in grouped_patients]
            labels = [data['labels'] for data in grouped_patients]
            batch_data.labels = torch.cat(labels, dim=0)

        else:
            set_trans_label = [data['set_trans_label'].unsqueeze(0) for data in grouped_patients]
            gru_label = [data['gru_label'].unsqueeze(0) for data in grouped_patients]

            batch_data.set_trans_label = torch.cat(set_trans_label, dim=0)
            batch_data.gru_label = torch.cat(gru_label, dim=0)

        if 'labels_additional' in grouped_patients[0].keys():
            labels_additional = [data['labels_additional'] for data in grouped_patients]
            batch_data.labels_additional = torch.cat(labels_additional, dim=0)

        if self.get_additional:
            _batch = []
            batch_patient = []
            patient_indicator = []

            for patient_idx, data in enumerate(grouped_patients):
                _visit_node_ids = data['visit_node_ids']
                _visit_edge_ids = data['visit_edge_ids']
                _visit_edge_index = data['visit_edge_index']
                _visit_edge_attr = data['visit_edge_attr']

                for single_visit_node_ids, single_visit_edge_ids, single_visit_edge_index, single_visit_edge_attr in zip(
                        _visit_node_ids, _visit_edge_ids, _visit_edge_index, _visit_edge_attr):

                    single_visit_graph = Data()
                    single_visit_graph.node_ids = torch.unique(single_visit_node_ids)  # Remove duplicate padding
                    single_visit_graph.edge_ids = single_visit_edge_ids
                    single_visit_graph.edge_index = self.reindex_edge_index(single_visit_edge_index)
                    single_visit_graph.edge_attr = single_visit_edge_attr

                    _batch.append(single_visit_graph)
                    batch_patient.append(patient_idx)
                    patient_indicator.extend([patient_idx] * len(single_visit_graph.node_ids))

            batch_graph = Batch.from_data_list(_batch)

            batch_data.cat_node_ids = batch_graph.node_ids
            batch_data.cat_edge_ids = batch_graph.edge_ids
            batch_data.cat_edge_index = batch_graph.edge_index
            batch_data.cat_edge_attr = batch_graph.edge_attr
            batch_data.batch = torch.tensor(patient_indicator)
            batch_data.visit_nodes = torch.cat(visit_nodes, dim=0)
            batch_data.ehr_nodes = torch.stack(ehr_nodes)
            batch_data.batch_patient = torch.tensor(batch_patient)

        else:
            batch_data.cat_node_ids = 'None'
            batch_data.cat_edge_ids = 'None'
            batch_data.cat_edge_index = 'None'
            batch_data.cat_edge_attr = 'None'
            batch_data.batch = 'None'
            batch_data.visit_nodes = 'None'
            batch_data.ehr_nodes = 'None'
            batch_data.visit_edge_ids = 'None'
            batch_data.batch_patient = 'None'

        return batch_data
