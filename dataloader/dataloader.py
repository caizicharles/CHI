import torch
from torch_geometric.data import Data
import random


class DataLoader():

    def __init__(self, dataset: list, task: str, batch_size=1, shuffle=False):

        self.dataset = dataset
        self.task = task
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_index = 0

        if self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        random.shuffle(self.dataset.dataset)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __iter__(self):
        return self

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
        batch_data.edge_index = self.dataset.edge_index
        batch_data.edge_attr = self.dataset.edge_attr
        node_ids = [data['node_ids'] for data in grouped_patients]
        labels = [[data['y']] for data in grouped_patients]
        visit_rel_times = [data['visit_rel_times'] for data in grouped_patients]
        visit_order = [data['visit_order'] for data in grouped_patients]

        batch_data.node_ids = torch.stack(node_ids)
        batch_data.labels = torch.tensor(labels)
        batch_data.visit_rel_times = torch.stack(visit_rel_times)
        batch_data.visit_order = torch.stack(visit_order)

        return batch_data
