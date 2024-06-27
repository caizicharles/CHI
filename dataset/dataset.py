import os.path as osp
import numpy as np
import torch
from torch.utils import data
import itertools
import random
from tqdm import tqdm

from utils import *

import time
import multiprocessing as mp


def split_by_patient(patients: dict, ratio: list):

    patients_k = np.array(list(patients.keys()))

    patient_num = len(patients)
    indices = list(np.arange(patient_num))

    train_num = int(ratio[0] * patient_num)
    val_num = int(ratio[1] * patient_num)

    train_indices = sorted(random.sample(indices, k=train_num))
    remain_indices = list(set(indices) - set(train_indices))
    val_indices = sorted(random.sample(remain_indices, k=val_num))
    test_indices = sorted(list(set(remain_indices) - set(val_indices)))

    train_patients = {key: patients[key] for key in patients_k[train_indices]}
    val_patients = {key: patients[key] for key in patients_k[val_indices]}
    test_patients = {key: patients[key] for key in patients_k[test_indices]}

    return train_patients, val_patients, test_patients


class MIMICIVBaseDataset(data.Dataset):

    def __init__(
            self,
            patients,
            task,
            nodes,
            #  graph,
            triplets,
            diagnoses_map,
            procedures_map,
            prescriptions_map,
            nodes_in_visits,
            code_pad_dim=None,
            visit_pad_dim=None,
            dataset_path=None):

        self.task = task
        # self.graph = graph
        self.nodes = nodes
        self.triplets = triplets
        self.diagnoses_map = diagnoses_map
        self.procedures_map = procedures_map
        self.prescriptions_map = prescriptions_map
        self.nodes_in_visits = nodes_in_visits
        self.code_pad_dim = code_pad_dim
        self.visit_pad_dim = visit_pad_dim
        self.PAD_NAME = 'N/A'

        if dataset_path is not None:
            assert osp.isfile(osp.join(dataset_path, 'padded_mimiciv.pickle')),\
                'padded dataset file is invalid.'
            self.dataset = read_pickle_file(dataset_path, 'padded_mimiciv.pickle')

        else:
            self.dataset = self.organize_patients(patients, task)

            if code_pad_dim is not None and visit_pad_dim is not None:
                self.dataset = self.pad_dataset(self.dataset, code_pad_dim, visit_pad_dim)
                self.dataset = self.construct_dataset(dataset=self.dataset,
                                                      nodes=nodes,
                                                      triplets=triplets,
                                                      nodes_in_visits=nodes_in_visits)
                # save_with_pickle(self.dataset, '/home/engs2635/Desktop/caizi/CHI/data/mimiciv', 'padded_mimiciv.pickle')

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_edges(self):
        return len(self.triplets)

    def find_code(self, code_name: str, table: str):

        assert table in ["diagnoses", "procedures", "prescriptions"], 'Unsupported table type: {table}'

        if table == 'diagnoses':
            return self.diagnoses_map[1][code_name]
        elif table == 'procedures':
            return self.procedures_map[1][code_name]
        elif table == 'prescriptions':
            return self.prescriptions_map[1][code_name]

    def find_code_name(self, code: str, table: str):

        assert table in ["diagnoses", "procedures", "prescriptions"], 'Unsupported table type: {table}'

        if table == 'diagnoses':
            return self.diagnoses_map[0][code]
        elif table == 'procedures':
            return self.procedures_map[0][code]
        elif table == 'prescriptions':
            return self.prescriptions_map[0][code]

    def pad_dataset(self, dataset: list, code_pad_dim: int, visit_pad_dim: int):

        for patient in dataset:
            visit_codes = patient['x'].copy()

            for idx, codes in enumerate(visit_codes):
                if len(codes) < code_pad_dim:
                    pad_length = code_pad_dim - len(codes)
                    visit_codes[idx] = codes + [self.PAD_NAME] * pad_length

                elif len(codes) > code_pad_dim:
                    raise ValueError('code_pad_dim must exceed max code length')

            if len(visit_codes) < visit_pad_dim:
                pad_length = visit_pad_dim - len(visit_codes)

                for _ in range(pad_length):
                    visit_codes.append([self.PAD_NAME] * code_pad_dim)

            elif len(visit_codes) > visit_pad_dim:
                raise ValueError('visit_pad_dim must exceed max visit num')

            patient['x'] = visit_codes

        return dataset

    def mortality_prediction_fn(self, patient):

        patient_input = {
            'patient_id': None,
            'visit_ids': [],
            #  'visit_x_map': [],
            'x': [],
            'y': None
        }

        patient_input['patient_id'] = patient.patient_id

        for (i, visit) in enumerate(patient):
            patient_input['visit_ids'].append(visit.visit_id)
            codes = self.nodes_in_visits[visit.visit_id]
            # patient_input['visit_x_map'].append([i]*len(codes))
            patient_input['x'].append(codes)

            if i == len(patient) - 1:
                if visit.discharge_status not in [0, 1]:
                    patient_input['y'] = 0
                else:
                    patient_input['y'] = int(visit.discharge_status)

        # patient_input['visit_x_map'] = [val for sub in patient_input['visit_x_map'] for val in sub]
        # patient_input['x'] = [val for sub in patient_input['x'] for val in sub]

        return patient_input

    def readmission_prediction_fn(self, patient, time_window=15):

        patient_input = {
            'patient_id': None,
            'visit_ids': [],
            #  'visit_x_map': [],
            'x': [],
            'y': None
        }

        patient_input['patient_id'] = patient.patient_id

        for (i, visit) in enumerate(patient):
            patient_input['visit_ids'].append(visit.visit_id)
            codes = self.nodes_in_visits[visit.visit_id]
            # patient_input['visit_x_map'].append([i]*len(codes))
            patient_input['x'].append(codes)

            if i == len(patient) - 1:
                time_diff = (visit.encounter_time - patient[i - 1].encounter_time).days
                readmission_label = 1 if time_diff < time_window else 0
                patient_input['y'] = readmission_label

        # patient_input['visit_x_map'] = [val for sub in patient_input['visit_x_map'] for val in sub]
        # patient_input['x'] = [val for sub in patient_input['x'] for val in sub]

        return patient_input

    def organize_patients(self, patients, task):

        dataset = []

        if task == 'mortality_prediction':
            task_fn = self.mortality_prediction_fn

        elif task == 'readmission_prediction':
            task_fn = self.readmission_prediction_fn

        elif task == 'los_prediction':
            pass

        elif task == 'drug_recommendation':
            pass

        for patient in patients.values():
            dataset.append(task_fn(patient))

        return dataset

    def extract_visit_edges(self, nodes_in_visit, nodes, triplets):

        src_dst = triplets[:, [1, -1]]

        edge_indices = []
        src_indices = []
        dst_indices = []
        edge_loc = []

        pairwise_combinations = list(itertools.permutations(nodes_in_visit, 2))

        for pair in pairwise_combinations:
            loc = np.where(np.all(src_dst == pair, axis=1))[0][0]
            edge_loc.append(loc)
            src_indices.append(np.where(nodes == pair[0])[0][0])
            dst_indices.append(np.where(nodes == pair[1])[0][0])

        edge_indices.append(src_indices)
        edge_indices.append(dst_indices)
        edges = triplets[edge_loc]
        edges = edges[:, 1:]

        return edges, edge_indices, edge_loc

    def construct_dataset(self, dataset, nodes, triplets, nodes_in_visits):

        nodes = np.array(nodes) if type(nodes) != np.ndarray else nodes
        triplets = np.array(triplets) if type(triplets) != np.ndarray else triplets

        for patient in dataset:
            visit_codes = patient['x']
            node_ids = []

            for codes in visit_codes:
                single_visit_ids = []

                for code in codes:
                    if code == self.PAD_NAME:
                        single_visit_ids.append(0)
                    else:
                        single_visit_ids.append(np.where(nodes == code)[0][0] + 1)

                node_ids.append(single_visit_ids)

            node_ids = torch.tensor(node_ids)
            patient['node_ids'] = node_ids

        src_list = []
        dst_list = []
        src_dst = triplets[:, [1, -1]]

        for src, dst in src_dst:
            src_list.append(np.where(nodes == src)[0][0])
            dst_list.append(np.where(nodes == dst)[0][0])

        self.edge_index = torch.tensor([src_list, dst_list])

        # node_ids = torch.tensor(node_ids) if type(node_ids) != torch.Tensor else node_ids
        # edge_ids = torch.tensor(edge_ids) if type(edge_ids) != torch.Tensor else edge_ids
        # edge_idx = torch.tensor(edge_idx) if type(edge_idx) != torch.Tensor else edge_idx
        # label = torch.tensor(label) if type(label) != torch.Tensor else label

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
        # return self.get_subset(index,
        #                        self.dataset,
        #                        self.nodes,
        #                        self.triplets,
        #                        self.nodes_in_visits)


# class MIMICIVBaseDataset(data.Dataset):

#     def __init__(self,
#                  dataset,
#                  task,
#                  nodes,
#                  graph,
#                  triplets,
#                  diagnoses_map,
#                  procedures_map,
#                  prescriptions_map,
#                  nodes_in_visits,
#                  pad_dim=None):

#         self.dataset = dataset
#         self.task = task
#         self.graph = graph
#         self.nodes = nodes
#         self.triplets = triplets
#         self.diagnoses_map = diagnoses_map
#         self.procedures_map = procedures_map
#         self.prescriptions_map = prescriptions_map
#         self.nodes_in_visits = nodes_in_visits
#         self.pad_dim = pad_dim
#         self.PAD_ID = -1

#         self.x = []

#     @property
#     def num_nodes(self):
#         return len(self.nodes)

#     @property
#     def num_edges(self):
#         return len(self.triplets)

#     def find_code(self, code_name: str, table: str):

#         assert table in ["diagnoses", "procedures", "prescriptions"], 'Unsupported table type: {table}'

#         if table == 'diagnoses':
#             return self.diagnoses_map[1][code_name]
#         elif table == 'procedures':
#             return self.procedures_map[1][code_name]
#         elif table == 'prescriptions':
#             return self.prescriptions_map[1][code_name]

#     def find_code_name(self, code: str, table: str):

#         assert table in ["diagnoses", "procedures", "prescriptions"], 'Unsupported table type: {table}'

#         if table == 'diagnoses':
#             return self.diagnoses_map[0][code]
#         elif table == 'procedures':
#             return self.procedures_map[0][code]
#         elif table == 'prescriptions':
#             return self.prescriptions_map[0][code]

#     def extract_visit_edges(self, nodes_in_visit, nodes, triplets):

#         src_dst = triplets[:, [1,-1]]

#         edge_indices = []
#         edge_loc = []

#         pairwise_combinations = list(itertools.permutations(nodes_in_visit, 2))
#         for pair in pairwise_combinations:
#             loc = np.where(np.all(src_dst == pair, axis=1))[0][0]
#             edge_loc.append(loc)

#             src_idx = np.where(nodes == pair[0])[0][0]
#             dst_idx = np.where(nodes == pair[1])[0][0]
#             edge_indices.append([[src_idx], [dst_idx]])

#         edges = triplets[edge_loc]
#         edges = edges[:, 1:]

#         return edges, edge_indices, edge_loc

#     def get_subset(self, index, dataset, nodes, triplets, nodes_in_visits, pad_dim=None):

#         nodes = np.array(nodes) if type(nodes) != np.ndarray else nodes
#         triplets = np.array(triplets) if type(triplets) != np.ndarray else triplets

#         visit_data = dataset[index]
#         visit_id = visit_data['visit_id']
#         label = visit_data['label']

#         node_ids = []
#         node_types = nodes_in_visits[visit_id]
#         for node_type in node_types:
#             node_ids.append(np.where(nodes == node_type)[0][0])

#         if pad_dim is not None:
#             pad_length = pad_dim - len(node_ids)

#             for _ in range(pad_length):
#                 node_ids.append(self.PAD_ID)

#         edges, edge_idx, edge_ids = self.extract_visit_edges(node_types, nodes, triplets)

#         node_ids = torch.tensor(node_ids) if type(node_ids) != torch.Tensor else node_ids
#         edge_ids = torch.tensor(edge_ids) if type(edge_ids) != torch.Tensor else edge_ids
#         edge_idx = torch.tensor(edge_idx) if type(edge_idx) != torch.Tensor else edge_idx
#         label = torch.tensor(label) if type(label) != torch.Tensor else label

#         # print(node_ids)
#         self.x.append(len(edge_ids))

#         return {
#             # 'nodes': node_types,
#                 'node_ids': node_ids,
#                 # 'edges': edges,
#                 'edge_ids': edge_ids,
#                 'edge_idx': edge_idx,
#                 'label': label}

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         return self.get_subset(index,
#                                self.dataset,
#                                self.nodes,
#                                self.triplets,
#                                self.nodes_in_visits,
#                                self.pad_dim)
