import os.path as osp
from pyhealth.datasets import MIMIC4Dataset
import pyhealth.medcode as pymed
import pyhealth.tasks as pytasks
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import *


def preprocess_dataset(raw_dataset_path: str,
                       dataset_name: str,
                       patient_info: dict,
                       save_path=None,
                       filtered=True,
                       filter_params={
                           'age_thresh_low': 18,
                           'age_thresh_high': 80,
                           'code_thresh': 100
                       }):

    dataset_path = osp.join(raw_dataset_path, dataset_name)
    assert osp.isdir(dataset_path), f'Raw dataset path: {dataset_path} is invalid.'

    if dataset_name == "mimiciv":
        if filtered:
            filtered_dataset_path = osp.join(save_path, 'filtered_mimiciv.pickle')

            if osp.exists(filtered_dataset_path):
                return read_pickle_file(save_path, 'filtered_mimiciv.pickle')

        processed_dataset_path = osp.join(save_path, f'mimiciv.pickle')

        if not osp.exists(processed_dataset_path):
            dataset_path = osp.join(dataset_path, '2.2/hosp/')
            dataset = MIMIC4Dataset(root=dataset_path,
                                    tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                                    code_mapping={
                                        "NDC": ("ATC", {
                                            "target_kwargs": {
                                                "level": 3
                                            }
                                        }),
                                        "ICD9CM": "CCSCM",
                                        "ICD9PROC": "CCSPROC",
                                        "ICD10CM": "CCSCM",
                                        "ICD10PROC": "CCSPROC"
                                    },
                                    dev=False,
                                    refresh_cache=True)
            '''
            # code_nums = get_dataset_table_stat(dataset)

            # ccscm = pymed.InnerMap.load('CCSCM')
            # ccsproc = pymed.InnerMap.load('CCSPROC')
            # atc = pymed.InnerMap.load('ATC')

            # for patient in dataset.patients.values():
            #     for visit in patient:
                #     diagnosis_list = visit.get_code_list('diagnoses_icd')
                #     procedure_list = visit.get_code_list('procedures_icd')
                    # prescription_list = visit.get_code_list('prescriptions')
                    
                #     for idx, diagnosis in enumerate(diagnosis_list):
                #         diagnosis_list[idx] = ccscm.lookup(diagnosis)
                #     for idx, procedure in enumerate(procedure_list):
                #         procedure_list[idx] = ccsproc.lookup(procedure)
                #     for idx, prescription in enumerate(prescription_list):
                #         prescription_list[idx] = atc.lookup(prescription)

                #     print(diagnosis_list)
                #     print(procedure_list)
                #     print(prescription_list)
                #     exit()

            # tally_set = get_dataset_visit_freq(dataset)
            '''

        else:
            dataset = read_pickle_file(save_path, dataset_name)

        patients = dataset.patients

        if filtered:
            patients = filter_dataset(patients=patients,
                                      patient_info=patient_info,
                                      age_thresh_low=filter_params['age_thresh_low'],
                                      age_thresh_high=filter_params['age_thresh_high'],
                                      code_thresh=filter_params['code_thresh'],
                                      visit_thresh=filter_params['visit_thresh'])

    return patients


def filter_dataset(patients: dict,
                   patient_info: dict,
                   age_thresh_low=18,
                   age_thresh_high=80,
                   code_thresh=100,
                   visit_thresh=15):

    filtered_patients = patients.copy()

    for (id, patient) in tqdm(patients.items()):
        pos = patient_info['subject_id'].index(id)
        age = int(patient_info['anchor_age'][pos])

        if age < age_thresh_low or age > age_thresh_high:
            del filtered_patients[id]
            continue

        if len(patient) <= 1 or len(patient) > visit_thresh:
            del filtered_patients[id]
            continue

        for visit in patient:
            num_codes = 0
            num_codes += len(visit.get_code_list('diagnoses_icd'))
            num_codes += len(visit.get_code_list('procedures_icd'))
            num_codes += len(visit.get_code_list('prescriptions'))

            if num_codes > code_thresh:
                del filtered_patients[id]
                break

    return filtered_patients


# def set_dataset_task(dataset, task: str):

#     assert task in ['mortality_prediction', 'readmission_prediction', 'los_prediction', 'drug_recommendation'],\
#     'Task is invalid.'

#     if dataset.dataset_name == 'MIMIC4Dataset':
#         if task == 'mortality_prediction':
#             tasked_dataset = dataset.set_task(pytasks.mortality_prediction_mimic4_fn)
#         elif task == 'readmission_prediction':
#             tasked_dataset = dataset.set_task(pytasks.readmission_prediction_mimic4_fn)
#         elif task == 'los_prediction':
#             tasked_dataset = dataset.set_task(pytasks.length_of_stay_prediction_mimic4_fn)
#         elif task == 'drug_recommendation':
#             tasked_dataset = dataset.set_task(pytasks.drug_recommendation_mimic4_fn)

#     return tasked_dataset


def get_dataset_table_stat(dataset):

    diagnosis_list = []
    procedure_list = []
    prescription_list = []

    for patient in dataset.patients.values():
        for visit in patient:
            diagnosis_list.append(visit.get_code_list('diagnoses_icd'))
            procedure_list.append(visit.get_code_list('procedures_icd'))
            prescription_list.append(visit.get_code_list('prescriptions'))

    d_flat = [d for sub in diagnosis_list for d in sub]
    pro_flat = [pro for sub in procedure_list for pro in sub]
    pre_flat = [pre for sub in prescription_list for pre in sub]

    d_flat = list(dict.fromkeys(d_flat))
    pro_flat = list(dict.fromkeys(pro_flat))
    pre_flat = list(dict.fromkeys(pre_flat))

    return len(d_flat), len(pro_flat), len(pre_flat)


def get_dataset_visit_freq(dataset):

    tally = []

    for patient in dataset.patients.values():
        tally.append(len(patient))

    num, freq = np.unique(tally, return_counts=True)

    return num, freq


def gen_NDC_ATC_mapping(dataset):

    p = []

    for patient in dataset.patients.values():
        for visit in patient:
            p.append(visit.get_code_list('prescriptions'))

    p_ = [e for sub in p for e in sub]

    p_ = list(dict.fromkeys(p_))
    map = {}
    mapping = pymed.CrossMap('NDC', 'ATC')
    atc = pymed.InnerMap.load('ATC')
    ndc = pymed.InnerMap.load('NDC')

    for element in p_:
        mapped_element = mapping.map(element, target_kwargs={"level": 3})
        map[element] = mapped_element

    new_map = {}

    for (k, v) in map.items():
        if v == []:
            word_v = 'ALL OTHER THERAPEUTIC PRODUCTS'
        else:
            word_v = atc.lookup(v[0])

        try:
            word_k = ndc.lookup(k)
            new_map[word_k] = word_v
        except:
            new_map['0.9 MG/ML Sodium Chloride'] = word_v

    return new_map
