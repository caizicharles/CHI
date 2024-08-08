import random
import os.path as osp
from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import itertools
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from utils.args import get_args
from utils.utils import *


def seed_everything(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_dependent_files(raw_data_path: str, processed_data_path: str, dataset_name: str):

    patient_info = read_csv_file(raw_data_path, '2.2/hosp/patients.csv')
    diagnoses_map = read_csv_file(processed_data_path, 'diagnoses_code.csv')
    procedures_map = read_csv_file(processed_data_path, 'procedures_code.csv')
    prescriptions_map = read_csv_file(processed_data_path, 'prescriptions_code.csv')
    node_id_to_name = read_pickle_file(processed_data_path, f'{dataset_name}_node_id_to_name.pickle')
    node_name_to_id = read_pickle_file(processed_data_path, f'{dataset_name}_node_name_to_id.pickle')
    visit_id_to_nodes = read_pickle_file(processed_data_path, f'{dataset_name}_visit_id_to_nodes.pickle')

    return {
        'node_id_to_name': node_id_to_name,
        'node_name_to_id': node_name_to_id,
        'patient_info': patient_info,
        'diagnoses_map': diagnoses_map,
        'procedures_map': procedures_map,
        'prescriptions_map': prescriptions_map,
        'visit_id_to_nodes': visit_id_to_nodes
    }


def get_dataset(raw_data_path: str, processed_data_path: str, dataset_name: str, save_dataset: bool = False):

    if osp.exists(osp.join(processed_data_path, f'{dataset_name}.pickle')):
        dataset = read_pickle_file(processed_data_path, dataset_name)

    else:
        if dataset_name == "mimiciv":
            raw_data_path = osp.join(raw_data_path, '2.2/hosp/')
            dataset = MIMIC4Dataset(root=raw_data_path,
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

        elif dataset_name == 'mimiciii':
            dataset = MIMIC3Dataset(root=raw_data_path,
                                    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
                                    code_mapping={
                                        "NDC": ("ATC", {
                                            "target_kwargs": {
                                                "level": 3
                                            }
                                        }),
                                        "ICD9CM": "CCSCM",
                                        "ICD9PROC": "CCSPROC"
                                    },
                                    dev=False,
                                    refresh_cache=True)

    if save_dataset:
        save_with_pickle(dataset, processed_data_path, f'{dataset_name}.pickle')

    return dataset


def filter_dataset(patients: dict,
                   patient_info: dict,
                   filter_args: dict,
                   processed_data_path: str,
                   dataset_name: str,
                   save_dataset: bool = False):

    if osp.exists(osp.join(processed_data_path, f'{dataset_name}_filtered.pickle')):
        filtered_patients = read_pickle_file(processed_data_path, f'{dataset_name}_filtered.pickle')

    else:
        age_thresh_low = filter_args['age_thresh_low']
        age_thresh_high = filter_args['age_thresh_high']
        code_thresh = filter_args['code_thresh']
        visit_thresh = filter_args['visit_thresh']

        filtered_patients = patients.copy()

        for (id, patient) in tqdm(patients.items(), desc='Filtering dataset'):
            pos = patient_info['subject_id'].index(id)
            age = int(patient_info['anchor_age'][pos])

            if age < age_thresh_low or age > age_thresh_high:
                del filtered_patients[id]
                continue

            if len(patient) <= 1 or len(patient) > visit_thresh:
                del filtered_patients[id]
                continue

            for visit in patient:
                num_diagnoses = len(visit.get_code_list('diagnoses_icd'))
                num_procedures = len(visit.get_code_list('procedures_icd'))
                num_prescriptions = len(visit.get_code_list('prescriptions'))
                num_codes = num_diagnoses + num_procedures + num_prescriptions

                # if num_diagnoses == 0 or num_procedures == 0 or num_prescriptions == 0:
                #     del filtered_patients[id]
                #     break

                if num_codes > code_thresh or num_codes == 0:
                    del filtered_patients[id]
                    break

        print(f'# of patients: {len(patients)} -> {len(filtered_patients)}')

    if save_dataset:
        save_with_pickle(filtered_patients, processed_data_path, f'{dataset_name}_filtered.pickle')

    return filtered_patients


def get_triplets(triplet_method: str,
                 triplet_args: dict,
                 node_name_to_id: dict,
                 node_maps: tuple,
                 patients: dict,
                 processed_data_path: str,
                 dataset_name: str,
                 save_triplets: bool = False):

    assert triplet_method in ['LLM', 'co-occurrence'], f'Unsupported triplet_mthod: {triplet_method}'

    if osp.exists(
            osp.join(processed_data_path, f'{args.dataset}_{args.triplet_method["name"]}_triplet_id_to_info.pickle')):
        triplet_id_to_info = read_pickle_file(
            processed_data_path, f'{args.dataset}_{args.triplet_method["name"]}_triplet_id_to_info.pickle')
    else:
        if triplet_method == 'LLM':
            # TODO
            pass

        elif triplet_method == 'co-occurrence':
            threshold = triplet_args['threshold']
            co_occurrence_freq = np.zeros((len(node_name_to_id), len(node_name_to_id)))

            for patient in tqdm(patients.values(), desc='Triplets'):
                for visit in patient:
                    occurred_codes = []

                    # Retrieve Codes in Visit
                    diagnoses_codes = visit.get_code_list('diagnoses_icd')
                    procedure_codes = visit.get_code_list('procedures_icd')
                    presciption_codes = visit.get_code_list('prescriptions')

                    # Remove Duplicates
                    diagnoses_codes = list(dict.fromkeys(diagnoses_codes))
                    procedure_codes = list(dict.fromkeys(procedure_codes))
                    presciption_codes = list(dict.fromkeys(presciption_codes))

                    # Code to code_name
                    for d in diagnoses_codes:
                        occurred_codes.append(node_maps[0][d])
                    for pro in procedure_codes:
                        occurred_codes.append(node_maps[1][pro])
                    for pre in presciption_codes:
                        occurred_codes.append(node_maps[2][pre])

                    # code_name to Index
                    for idx, code in enumerate(occurred_codes):
                        occurred_codes[idx] = node_name_to_id[code] - 1

                    if len(occurred_codes) > 1:
                        pairwise_combinations = list(itertools.permutations(occurred_codes, 2))
                        rows, cols = zip(*pairwise_combinations)
                        co_occurrence_freq[np.array(rows), np.array(cols)] += 1

            co_occurrence_idx = np.where(co_occurrence_freq >= threshold)
            freq = co_occurrence_freq[co_occurrence_idx[0], co_occurrence_idx[1]]

            node_names = np.array(list(node_name_to_id.keys()))
            src = node_names[co_occurrence_idx[0]]
            dst = node_names[co_occurrence_idx[1]]

            triplet_id_to_info = {}
            for idx in range(len(src)):
                triplet_id_to_info[idx + 1] = [src[idx], freq[idx], dst[idx]]

        if save_triplets:
            save_with_pickle(triplet_id_to_info, processed_data_path,
                             f'{dataset_name}_{triplet_method}_triplet_id_to_info.pickle')

    return triplet_id_to_info


def construct_graph(node_id_to_name: dict,
                    node_name_to_id: dict,
                    triplet_id_to_info: dict,
                    processed_data_path: str,
                    dataset_name: str,
                    triplet_method: str,
                    save_graph: bool = False,
                    save_triplet_maps: bool = False):

    if osp.exists(osp.join(processed_data_path, f'{dataset_name}_{triplet_method}_graph.pickle')):
        graph = read_pickle_file(processed_data_path, f'{dataset_name}_{triplet_method}_graph.pickle')

    else:
        x = torch.tensor(list(node_id_to_name.keys())).unsqueeze(-1)  # (node_num,1)

        triplet_info = np.array(list(triplet_id_to_info.values()))
        src_dst = triplet_info[:, [0, -1]]

        edge_src = []
        edge_dst = []
        for src, dst in src_dst:
            edge_src.append(node_name_to_id[src])
            edge_dst.append(node_name_to_id[dst])

        edge_index = [edge_src, edge_dst]
        edge_index = torch.tensor(edge_index)  # (2,edge_num)

        edge_attr = triplet_info[:, 1]
        edge_attr = list(map(float, edge_attr))
        edge_attr = torch.tensor(edge_attr).unsqueeze(-1)  # (edge_num,1)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        if save_graph:
            save_with_pickle(graph, processed_data_path, f'{dataset_name}_{triplet_method}_graph.pickle')

    if osp.exists(osp.join(processed_data_path,
                           f'{dataset_name}_{triplet_method}_triplet_id_to_edge_index')) and osp.exists(
                               processed_data_path, f'{dataset_name}_{triplet_method}_triplet_edge_index_to_id'):
        triplet_id_to_edge_index = read_pickle_file(processed_data_path,
                                                    f'{dataset_name}_{triplet_method}_triplet_id_to_edge_index')
        triplet_edge_index_to_id = read_pickle_file(processed_data_path,
                                                    f'{dataset_name}_{triplet_method}_triplet_edge_index_to_id')

    else:
        triplet_info = np.array(list(triplet_id_to_info.values()))
        src_dst = triplet_info[:, [0, -1]]

        edge_src = []
        edge_dst = []
        for src, dst in src_dst:
            edge_src.append(node_name_to_id[src])
            edge_dst.append(node_name_to_id[dst])

        edge_index = [edge_src, edge_dst]
        edge_index = torch.tensor(edge_index)  # (2,edge_num)

        triplet_id_to_edge_index = {}
        triplet_edge_index_to_id = {}

        for idx, pair in enumerate(edge_index.t()):
            triplet_id_to_edge_index[idx + 1] = pair.tolist()
            triplet_edge_index_to_id[str(pair.tolist())] = idx + 1

        if save_triplet_maps:
            save_with_pickle(triplet_id_to_edge_index, processed_data_path,
                             f'{dataset_name}_{triplet_method}_triplet_id_to_edge_index')
            save_with_pickle(triplet_edge_index_to_id, processed_data_path,
                             f'{dataset_name}_{triplet_method}_triplet_edge_index_to_id')

    return graph, triplet_id_to_edge_index, triplet_edge_index_to_id


def construct_patient(patient, patient_data: dict, graph: Data, visit_id_to_nodes: dict, node_name_to_id: dict,
                      triplet_edge_index_to_id: dict, visit_pad_dim: int):

    patient_data['patient_id'] = patient.patient_id
    subset_threshold = torch.max(graph.edge_index)

    visit_encounters = []
    visit_discharges = []

    for visit in patient:
        codes = visit_id_to_nodes[visit.visit_id]
        patient_data['visit_ids'].append(visit.visit_id)
        visit_encounters.append(visit.encounter_time)
        visit_discharges.append(visit.discharge_time)

        single_visit_node_ids = []
        single_visit_edge_ids = []
        single_visit_mask = []

        for code in codes:
            single_visit_node_ids.append(node_name_to_id[code])

        subset = torch.tensor(single_visit_node_ids, dtype=int)
        subset = subset[subset <= subset_threshold]
        # TODO reindexing in edge index
        visit_edge_index, visit_edge_attr = subgraph(subset=subset,
                                                     edge_index=graph.edge_index,
                                                     edge_attr=graph.edge_attr,
                                                     relabel_nodes=False)

        single_visit_node_ids = torch.unique(visit_edge_index.flatten()).tolist()
        single_visit_mask = [False] * len(single_visit_node_ids)

        patient_data['visit_node_ids'].append(single_visit_node_ids)
        patient_data['node_padding_mask'].append(single_visit_mask)
        patient_data['visit_edge_index'].append(visit_edge_index)
        patient_data['visit_edge_attr'].append(visit_edge_attr)

        edge_pairs = visit_edge_index.t()
        for pair in edge_pairs:
            id = triplet_edge_index_to_id[str(pair.tolist())]
            single_visit_edge_ids.append(id)
        patient_data['visit_edge_ids'].append(single_visit_edge_ids)

    visit_node_ids = patient_data['visit_node_ids']
    visit_nodes = torch.zeros(len(visit_node_ids), len(node_name_to_id) + 1)
    for i in range(len(visit_node_ids)):
        visit_nodes[i, visit_node_ids[i]] = 1
    patient_data['visit_nodes'] = visit_nodes

    patient_node_ids = torch.tensor([id for id_set in visit_node_ids for id in id_set], dtype=int)
    patient_node_ids = torch.unique(patient_node_ids)
    ehr_nodes = torch.zeros(len(node_name_to_id) + 1)
    ehr_nodes[patient_node_ids] = 1
    patient_data['ehr_nodes'] = ehr_nodes

    visit_rel_times = []
    hist_time = 0
    # TODO week distribution
    for enc, dis in zip(visit_encounters, visit_discharges):
        if hist_time == 0:
            visit_rel_times.append(0)
        else:
            day_dif = (enc - hist_time).days
            week_dif = abs(day_dif) // 7
            visit_rel_times.append(week_dif + 1)

        hist_time = dis
    patient_data['visit_rel_times'] = visit_rel_times

    return patient_data


def organize_dataset(patients,
                     graph: Data,
                     visit_id_to_nodes: list,
                     node_name_to_id: dict,
                     triplet_edge_index_to_id: dict,
                     visit_pad_dim: int,
                     processed_data_path: str,
                     dataset_name: str,
                     save_dataset: bool = False):

    if osp.exists(osp.join(processed_data_path, f'{dataset_name}_organized.pickle')):
        dataset = read_pickle_file(processed_data_path, f'{dataset_name}_organized.pickle')

    else:
        dataset = {}

        PATIENT_TEMPLATE = {
            'patient_id': None,
            'visit_ids': [],
            'visit_node_ids': [],
            'visit_edge_ids': [],
            'visit_edge_index': [],
            'visit_edge_attr': [],
            'visit_rel_times': [],
            'visit_order': None,
            'visit_nodes': None,
            'ehr_nodes': None,
            'node_padding_mask': [],
            'label': None
        }

        for id, patient in tqdm(patients.items(), desc='Organizing patients'):
            patient_data = deepcopy(PATIENT_TEMPLATE)
            patient_data = construct_patient(patient=patient,
                                             patient_data=patient_data,
                                             graph=graph,
                                             visit_id_to_nodes=visit_id_to_nodes,
                                             node_name_to_id=node_name_to_id,
                                             triplet_edge_index_to_id=triplet_edge_index_to_id,
                                             visit_pad_dim=visit_pad_dim)
            dataset[id] = patient_data

        if save_dataset:
            save_with_pickle(dataset, processed_data_path, f'{dataset_name}_organized.pickle')

    return dataset


def pad_dataset(dataset: dict,
                code_pad_dim: int,
                visit_pad_dim: int,
                PAD_ID: int,
                processed_data_path: str,
                dataset_name: str,
                save_dataset: bool = False):

    if osp.exists(osp.join(processed_data_path, f'{dataset_name}_padded.pickle')):
        dataset = read_pickle_file(processed_data_path, f'{dataset_name}_padded.pickle')

    else:
        for patient in tqdm(dataset.values(), desc='Padding'):

            visit_node_ids = patient['visit_node_ids']
            visit_edge_ids = patient['visit_edge_ids']
            node_padding_mask = patient['node_padding_mask']
            visit_edge_index = patient['visit_edge_index']
            visit_edge_attr = patient['visit_edge_attr']
            visit_rel_times = patient['visit_rel_times']
            visit_nodes = patient['visit_nodes']
            ehr_nodes = patient['ehr_nodes']

            # code padding and mask padding
            for idx, single_visit_ids in enumerate(visit_node_ids):
                if len(single_visit_ids) < code_pad_dim:
                    pad_length = code_pad_dim - len(single_visit_ids)
                    visit_node_ids[idx] = single_visit_ids + [PAD_ID] * pad_length
                    node_padding_mask[idx] += [True] * pad_length

                elif len(single_visit_ids) > code_pad_dim:
                    raise ValueError('code_pad_dim must exceed max code length')

            if len(visit_node_ids) < visit_pad_dim:
                pad_length = visit_pad_dim - len(visit_node_ids)

                for _ in range(pad_length):
                    visit_node_ids.append([PAD_ID] * code_pad_dim)
                    node_padding_mask.append([True] * code_pad_dim)

            elif len(visit_node_ids) > visit_pad_dim:
                raise ValueError('visit_pad_dim must exceed max visit num')

            # edge padding
            # edge_pad_dim = code_pad_dim**2

            # for idx, single_visit_ids in enumerate(visit_edge_ids):
            #     if len(single_visit_ids) < edge_pad_dim:
            #         pad_length = edge_pad_dim - len(single_visit_ids)
            #         visit_edge_ids[idx] = single_visit_ids + [PAD_ID] * pad_length

            #     elif len(single_visit_ids) > edge_pad_dim:
            #         raise ValueError('edge_pad_dim must exceed max code length')

            # if len(visit_edge_ids) < visit_pad_dim:
            #     pad_length = visit_pad_dim - len(visit_edge_ids)

            #     for _ in range(pad_length):
            #         visit_edge_ids.append([PAD_ID] * edge_pad_dim)

            # elif len(visit_edge_ids) > visit_pad_dim:
            #     raise ValueError('visit_pad_dim must exceed max visit num')

            visit_node_ids = torch.tensor(visit_node_ids,
                                          dtype=int) if not isinstance(visit_node_ids, torch.Tensor) else visit_node_ids
            

            for idx, (single_visit_edge_index,
                      single_visit_edge_attr) in enumerate(zip(visit_edge_index, visit_edge_attr)):
                if torch.eq(visit_node_ids[idx], PAD_ID).any():
                    visit_edge_ids[idx].append(PAD_ID)
                    visit_edge_index[idx] = torch.cat((torch.tensor([[PAD_ID], [PAD_ID]]), single_visit_edge_index), dim=-1)
                    visit_edge_attr[idx] = torch.cat((torch.tensor([[PAD_ID]], dtype=float), single_visit_edge_attr), dim=0)
                visit_edge_ids[idx] = torch.tensor(visit_edge_ids[idx], dtype=int)

            visit_rel_times = torch.tensor(
                visit_rel_times, dtype=int) if not isinstance(visit_rel_times, torch.Tensor) else visit_rel_times
            pad = torch.zeros(visit_pad_dim - len(visit_rel_times))
            visit_rel_times = torch.cat((visit_rel_times, pad), dim=0).to(torch.int)

            visit_order = torch.arange(visit_pad_dim, dtype=int)
            visit_nodes = torch.tensor(visit_nodes,
                                       dtype=int) if not isinstance(visit_nodes, torch.Tensor) else visit_nodes
            ehr_nodes = torch.tensor(ehr_nodes, dtype=int) if not isinstance(ehr_nodes, torch.Tensor) else ehr_nodes
            node_padding_mask = torch.tensor(
                node_padding_mask, dtype=bool) if not isinstance(node_padding_mask, torch.Tensor) else node_padding_mask

            patient['visit_node_ids'] = visit_node_ids
            patient['visit_edge_ids'] = visit_edge_ids
            patient['visit_edge_index'] = visit_edge_index
            patient['visit_edge_attr'] = visit_edge_attr
            patient['visit_rel_times'] = visit_rel_times
            patient['visit_order'] = visit_order
            patient['visit_nodes'] = visit_nodes
            patient['ehr_nodes'] = ehr_nodes
            patient['node_padding_mask'] = node_padding_mask

        if save_dataset:
            save_with_pickle(dataset, processed_data_path, f'{dataset_name}_padded.pickle')

    return dataset


def run(args):

    seed_everything(args.seed)

    raw_data_path = osp.join(args.raw_data_path, args.dataset)
    processed_data_path = osp.join(args.processed_data_path, args.dataset)

    file_lib = get_dependent_files(raw_data_path, processed_data_path, args.dataset)
    print('Got files')

    patient_info = format_from_csv(file_lib['patient_info'])
    diagnoses_code_to_name, diagnoses_name_to_code = format_code_map(file_lib['diagnoses_map'])
    procedures_code_to_name, procedures_name_to_code = format_code_map(file_lib['procedures_map'])
    prescriptions_code_to_name, prescriptions_name_to_code = format_code_map(file_lib['prescriptions_map'])

    dataset = get_dataset(raw_data_path=raw_data_path,
                          processed_data_path=processed_data_path,
                          dataset_name=args.dataset,
                          save_dataset=False)
    print('Got dataset')

    filtered_patients = filter_dataset(patients=dataset.patients,
                                       patient_info=patient_info,
                                       filter_args=args.dataset_filtering['args'],
                                       processed_data_path=processed_data_path,
                                       dataset_name=args.dataset,
                                       save_dataset=False)
    print('Filtered dataset')

    triplet_id_to_info = get_triplets(triplet_method=args.triplet_method['name'],
                                      triplet_args=args.triplet_method['args'],
                                      node_name_to_id=file_lib['node_name_to_id'],
                                      node_maps=(diagnoses_code_to_name, procedures_code_to_name,
                                                 prescriptions_code_to_name),
                                      patients=filtered_patients,
                                      processed_data_path=processed_data_path,
                                      dataset_name=args.dataset,
                                      save_triplets=False)

    graph, triplet_id_to_edge_index, triplet_edge_index_to_id = construct_graph(
        node_id_to_name=file_lib['node_id_to_name'],
        node_name_to_id=file_lib['node_name_to_id'],
        triplet_id_to_info=triplet_id_to_info,
        processed_data_path=processed_data_path,
        dataset_name=args.dataset,
        triplet_method=args.triplet_method['name'],
        save_graph=False,
        save_triplet_maps=False)
    print('Graph constructed')

    dataset = organize_dataset(patients=filtered_patients,
                               graph=graph,
                               visit_id_to_nodes=file_lib['visit_id_to_nodes'],
                               node_name_to_id=file_lib['node_name_to_id'],
                               triplet_edge_index_to_id=triplet_edge_index_to_id,
                               visit_pad_dim=args.dataset_filtering['args']['visit_thresh'],
                               processed_data_path=processed_data_path,
                               dataset_name=args.dataset,
                               save_dataset=False)
    print('Organized dataset')

    padded_dataset = pad_dataset(dataset=dataset,
                                 code_pad_dim=args.dataset_filtering['args']['code_thresh'],
                                 visit_pad_dim=args.dataset_filtering['args']['visit_thresh'],
                                 PAD_ID=0,
                                 processed_data_path=processed_data_path,
                                 dataset_name=args.dataset,
                                 save_dataset=True)
    print('Padded dataset')
    print('Data preparation complete')


if __name__ == '__main__':
    args = get_args()
    run(args=args)
