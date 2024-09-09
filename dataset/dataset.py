import numpy as np
import torch
from torch.utils import data
from torch_geometric.data import Data

from utils.utils import *


class MIMICIVBaseDataset(data.Dataset):

    def __init__(self, patients: dict, filtered_patients: dict, task: str, graph: Data,
                 prescriptions_code_to_name: dict):

        self.task = task
        self.graph = graph
        self.prescriptions_code_to_name = prescriptions_code_to_name
        self.dataset = self.get_label(patients, filtered_patients, task)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def categorize_los(self, days: int):

        if days < 1:
            return 0
        elif 1 <= days <= 7:
            return days
        elif 7 < days <= 14:
            return 8
        else:
            return 9

    def mortality_prediction_fn(self, patient, filtered_patient):

        for (i, visit) in enumerate(filtered_patient):

            if i == len(filtered_patient) - 1:
                if visit.discharge_status not in [0, 1]:
                    patient['labels'] = torch.tensor([0.])
                else:
                    patient['labels'] = torch.tensor([[float(visit.discharge_status)]])

        return patient

    def readmission_prediction_fn(self, patient, filtered_patient, time_window: int = 15):

        for (i, visit) in enumerate(filtered_patient):

            if i == len(filtered_patient) - 1:
                time_diff = (visit.encounter_time - filtered_patient[i - 1].encounter_time).days
                readmission_label = 1. if time_diff < time_window else 0.
                patient['labels'] = torch.tensor([[readmission_label]])

        return patient

    def los_prediction_fn(self, patient, filtered_patient):

        for (i, visit) in enumerate(filtered_patient):

            if i == len(filtered_patient) - 1:
                los_days = (visit.discharge_time - visit.encounter_time).days
                patient['labels'] = torch.tensor([[self.categorize_los(los_days)]])

        return patient

    def drug_recommendation_fn(self, patient, filtered_patient):

        label = torch.zeros(len(self.prescriptions_code_to_name))
        full_drug_names = list(self.prescriptions_code_to_name.values())
        last_visit_drug_names = patient['last_visit_drug_names']

        for name in last_visit_drug_names:
            label[full_drug_names.index(name)] = 1

        patient['labels'] = label.unsqueeze(0)

        return patient

    def pretrain_fn(self, patient):

        # num_nodes = self.graph.x.size(0)
        # visit_node_ids = patient['visit_node_ids']

        # history = torch.zeros(num_nodes, dtype=int)
        # label_1 = torch.zeros(visit_node_ids.size(0), num_nodes, dtype=int)
        # label_2 = torch.zeros(visit_node_ids.size(0), num_nodes, dtype=int)

        # for idx, single_visit_nodes in enumerate(visit_node_ids):
        #     unique_nodes = torch.unique(single_visit_nodes)
        #     label_1[idx][unique_nodes] = 1
        #     label_2[idx] = label_1[idx] | history
        #     history = label_2[idx]

        # patient['labels'] = label_1.unsqueeze(0).to(float)
        # patient['labels_additional'] = label_2.unsqueeze(0).to(float)

        return patient

    def get_label(self, patients: dict, filtered_patients: dict, task: str):

        dataset = []

        if task == 'pretrain':
            task_fn = self.pretrain_fn

        elif task == 'mortality_prediction':
            task_fn = self.mortality_prediction_fn

        elif task == 'readmission_prediction':
            task_fn = self.readmission_prediction_fn

        elif task == 'los_prediction':
            task_fn = self.los_prediction_fn

        elif task == 'drug_recommendation':
            task_fn = self.drug_recommendation_fn

        if task == 'pretrain':
            for id, patient in patients.items():
                dataset.append(task_fn(patient))
        else:
            for id, patient in patients.items():
                dataset.append(task_fn(patient, filtered_patients[id]))

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
