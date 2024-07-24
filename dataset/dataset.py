import os.path as osp
import numpy as np
import torch
from torch.utils import data
import itertools
import random
from tqdm import tqdm

from utils.utils import *
from utils.misc import str_to_datetime


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
            triplet_method,
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
        self.triplet_method = triplet_method
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
                                                      triplet_method=triplet_method,
                                                      visit_pad_dim=visit_pad_dim)
                # save_with_pickle(self.dataset, '/home/engs2635/Desktop/caizi/CHI/data/mimiciv', 'ready_mimiciv.pickle')

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_edges(self):
        return len(self.triplets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

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

    def mortality_prediction_fn(self, patient):

        patient_input = {
            'patient_id': None,
            'visit_ids': [],
            'node_ids': None,
            'edge_ids': None,
            'visit_encounters': [],
            'visit_discharges': [],
            'visit_node': None,
            'edge_names': [],
            'ehr_nodes': [],
            'padding_mask': [],
            'x': [],
            'y': None,
        }

        patient_input['patient_id'] = patient.patient_id

        for (i, visit) in enumerate(patient):
            patient_input['visit_ids'].append(visit.visit_id)
            patient_input['visit_encounters'].append(visit.encounter_time)
            patient_input['visit_discharges'].append(visit.discharge_time)
            codes = self.nodes_in_visits[visit.visit_id]
            patient_input['x'].append(codes)

            edges = itertools.permutations(codes, 2)
            edges = [list(pair) for pair in edges]
            patient_input['edge_names'].append(edges)

            if i == len(patient) - 1:
                if visit.discharge_status not in [0, 1]:
                    patient_input['y'] = 0.
                else:
                    patient_input['y'] = float(visit.discharge_status)

        return patient_input

    def readmission_prediction_fn(self, patient, time_window=15):

        patient_input = {
            'patient_id': None,
            'visit_ids': [],
            'node_ids': None,
            'edge_ids': None,
            'visit_encounters': [],
            'visit_discharges': [],
            'visit_node': None,
            'ehr_nodes': [],
            'padding_mask': [],
            'x': [],
            'y': None,
        }

        patient_input['patient_id'] = patient.patient_id

        for (i, visit) in enumerate(patient):
            patient_input['visit_ids'].append(visit.visit_id)
            patient_input['visit_encounters'].append(visit.encounter_time)
            patient_input['visit_discharges'].append(visit.discharge_time)
            codes = self.nodes_in_visits[visit.visit_id]
            patient_input['x'].append(codes)

            if i == len(patient) - 1:
                time_diff = (visit.encounter_time - patient[i - 1].encounter_time).days
                readmission_label = 1. if time_diff < time_window else 0.
                patient_input['y'] = readmission_label

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

    def pad_dataset(self, dataset: list, code_pad_dim: int, visit_pad_dim: int):

        for patient in dataset:
            patient_codes = patient['x'].copy()
            patient_edges = patient['edge_names'].copy()

            # code padding
            for idx, codes in enumerate(patient_codes):
                if len(codes) < code_pad_dim:
                    pad_length = code_pad_dim - len(codes)
                    patient_codes[idx] = codes + [self.PAD_NAME] * pad_length

                elif len(codes) > code_pad_dim:
                    raise ValueError('code_pad_dim must exceed max code length')

            if len(patient_codes) < visit_pad_dim:
                pad_length = visit_pad_dim - len(patient_codes)

                for _ in range(pad_length):
                    patient_codes.append([self.PAD_NAME] * code_pad_dim)

            elif len(patient_codes) > visit_pad_dim:
                raise ValueError('visit_pad_dim must exceed max visit num')
            
            # edge padding
            edge_pad_dim = code_pad_dim**2

            for idx, edges in enumerate(patient_edges):
                if len(edges) < edge_pad_dim:
                    pad_length = edge_pad_dim - len(edges)
                    patient_edges[idx] = edges + [self.PAD_NAME] * pad_length

                elif len(edges) > edge_pad_dim:
                    raise ValueError('edge_pad_dim must exceed max code length')
                
            if len(patient_edges) < visit_pad_dim:
                pad_length = visit_pad_dim - len(patient_edges)

                for _ in range(pad_length):
                    patient_edges.append([self.PAD_NAME] * edge_pad_dim)

            elif len(patient_edges) > visit_pad_dim:
                raise ValueError('visit_pad_dim must exceed max visit num')

            patient['x'] = patient_codes
            patient['edge_names'] = patient_edges

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

    def construct_dataset(self, dataset, nodes, triplets, triplet_method, visit_pad_dim):

        nodes = np.array(nodes) if type(nodes) != np.ndarray else nodes
        triplets = np.array(triplets) if type(triplets) != np.ndarray else triplets
        src_dst = triplets[:, [1, -1]]

        for patient in tqdm(dataset):
            # node_ids, edge_ids, and padding_mask
            patient_codes = patient['x']
            patient_edges = patient['edge_names']
            node_ids = []
            edge_ids = []
            padding_mask = []

            for codes in patient_codes:
                single_visit_node_ids = []
                single_visit_mask = []

                for code in codes:
                    if code == self.PAD_NAME:
                        single_visit_node_ids.append(0)
                        single_visit_mask.append(True)
                    else:
                        single_visit_node_ids.append(np.where(nodes == code)[0][0] + 1)
                        single_visit_mask.append(False)

                node_ids.append(single_visit_node_ids)
                padding_mask.append(single_visit_mask)

            for edges in patient_edges:
                single_visit_edge_ids = []

                for edge in edges:
                    if edge == self.PAD_NAME:
                        single_visit_edge_ids.append(0)
                    else:
                        if edge in src_dst:
                            single_visit_edge_ids.append(np.where(src_dst == edge)[0][0] + 1)
                        else:
                            single_visit_edge_ids.append(0)

                edge_ids.append(single_visit_edge_ids)

            node_ids = torch.tensor(node_ids)
            padding_mask = torch.tensor(padding_mask)
            edge_ids = torch.tensor(edge_ids)

            patient['node_ids'] = node_ids
            patient['padding_mask'] = padding_mask
            patient['edge_ids'] = edge_ids

            # visit_node
            visit_node = torch.zeros(visit_pad_dim, len(nodes)+1)
            for i in range(node_ids.size(0)):
                visit_node[i, node_ids[i]] = 1
            patient['visit_node'] = visit_node

            # visit_rel_times and visit_order
            encounters = patient['visit_encounters']
            discharges = patient['visit_discharges']
            visit_rel_times = []
            hist_time = 0

            for enc, dis in zip(encounters, discharges):
                if hist_time == 0:
                    visit_rel_times.append(0)

                else:
                    day_dif = (enc - hist_time).days
                    week_dif = abs(day_dif) // 7
                    visit_rel_times.append(week_dif + 1)

                hist_time = dis

            visit_rel_times = torch.tensor(visit_rel_times)
            pad = torch.zeros(len(patient_codes) - len(visit_rel_times))
            visit_rel_times = torch.cat((visit_rel_times, pad), dim=0).to(torch.int)
            visit_order = torch.arange(len(patient_codes), dtype=int)
            patient['visit_rel_times'] = visit_rel_times
            patient['visit_order'] = visit_order

        src_list = []
        dst_list = []

        # edge_attr
        if triplet_method == 'co-occurrence':
            edge_weights = triplets[:, 2].astype(float)
            self.edge_attr = torch.tensor(edge_weights)
        else:
            relations = triplets[:, 2]
            # TODO

        # edge_index
        for src, dst in src_dst:
            src_list.append(np.where(nodes == src)[0][0] + 1)
            dst_list.append(np.where(nodes == dst)[0][0] + 1)

        self.edge_index = torch.tensor([src_list, dst_list])

        return dataset


def split_by_patient(patients: dict, ratio: list):

    patient_num = len(patients)
    patients_k = np.array(list(patients.keys()))
    np.random.shuffle(patients_k)

    train_num = int(ratio[0] * patient_num)
    val_num = int(ratio[1] * patient_num)

    train_k = patients_k[:train_num]
    val_k = patients_k[train_num:train_num + val_num]
    test_k = patients_k[train_num + val_num:]

    train_patients = {key: patients[key] for key in train_k}
    val_patients = {key: patients[key] for key in val_k}
    test_patients = {key: patients[key] for key in test_k}

    return train_patients, val_patients, test_patients
