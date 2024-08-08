import numpy as np
import torch
from torch.utils import data
from torch_geometric.data import Data

from utils.utils import *


class MIMICIVBaseDataset(data.Dataset):

    def __init__(self, patients: dict, filtered_patients: dict, task: str, graph: Data):

        self.dataset = self.get_label(patients, filtered_patients, task)
        self.task = task
        self.graph = graph

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def mortality_prediction_fn(self, patient, filtered_patient):

        for (i, visit) in enumerate(filtered_patient):

            if i == len(filtered_patient) - 1:
                if visit.discharge_status not in [0, 1]:
                    patient['labels'] = torch.tensor([0.])
                else:
                    patient['labels'] = torch.tensor([float(visit.discharge_status)])

        return patient

    def readmission_prediction_fn(self, patient, filtered_patient, time_window: int = 15):

        for (i, visit) in enumerate(filtered_patient):

            if i == len(filtered_patient) - 1:
                time_diff = (visit.encounter_time - filtered_patient[i - 1].encounter_time).days
                readmission_label = 1. if time_diff < time_window else 0.
                patient['labels'] = torch.tensor([readmission_label])

        return patient

    def get_label(self, patients: dict, filtered_patients: dict, task: str):

        dataset = []

        if task == 'mortality_prediction':
            task_fn = self.mortality_prediction_fn

        elif task == 'readmission_prediction':
            task_fn = self.readmission_prediction_fn

        elif task == 'los_prediction':
            pass

        elif task == 'drug_recommendation':
            pass

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
