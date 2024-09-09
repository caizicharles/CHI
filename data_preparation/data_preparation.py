import random
import os
import os.path as osp
from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import itertools
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from sklearn.cluster import KMeans

from utils.args import get_args
from utils.utils import *
from llm_embedding import get_BERT_embeddings, verify_embeddings


def seed_everything(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dependent_files(raw_data_path: str, processed_data_path: str, dataset_name: str):

    if dataset_name == 'mimiciv':
        patient_info = read_csv_file(raw_data_path, '2.2/hosp/patients.csv')
    else:
        patient_info = read_csv_file(raw_data_path, 'PATIENTS.csv')

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
            if dataset_name == 'mimiciv':
                pos = patient_info['subject_id'].index(id)
                age = int(patient_info['anchor_age'][pos])

                if age < age_thresh_low or age > age_thresh_high:
                    del filtered_patients[id]
                    continue

            elif dataset_name == 'mimiciii':
                dob = patient.birth_datetime.year

                for visit in patient:
                    dov = visit.encounter_time.year
                    age = dov - dob

                    if age < age_thresh_low or age > age_thresh_high:
                        break

                if age < age_thresh_low or age > age_thresh_high:
                    del filtered_patients[id]
                    continue

            if len(patient) <= 1 or len(patient) > visit_thresh:
                del filtered_patients[id]
                continue

            for visit in patient:
                if dataset_name == 'mimiciv':
                    num_diagnoses = len(visit.get_code_list('diagnoses_icd'))
                    num_procedures = len(visit.get_code_list('procedures_icd'))
                    num_prescriptions = len(visit.get_code_list('prescriptions'))
                elif dataset_name == 'mimiciii':
                    num_diagnoses = len(visit.get_code_list('DIAGNOSES_ICD'))
                    num_procedures = len(visit.get_code_list('PROCEDURES_ICD'))
                    num_prescriptions = len(visit.get_code_list('PRESCRIPTIONS'))

                num_codes = num_diagnoses + num_procedures + num_prescriptions

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

    assert triplet_method in ['LLM', 'co-occurrence', 'random'], f'Unsupported triplet_mthod: {triplet_method}'

    if osp.exists(
            osp.join(processed_data_path,
                     f'{args.dataset}_{args.triplet_method["name"]}_triplet_id_to_info.pickle')) and osp.exists(
                         osp.join(processed_data_path,
                                  f'{args.dataset}_{args.triplet_method["name"]}_edge_attr.pickle')) and osp.exists(
                                      osp.join(processed_data_path,
                                               f'{args.dataset}_{args.triplet_method["name"]}_node_attr.pickle')):
        triplet_id_to_info = read_pickle_file(
            processed_data_path, f'{args.dataset}_{args.triplet_method["name"]}_triplet_id_to_info.pickle')
        node_attr = read_pickle_file(processed_data_path,
                                     f'{args.dataset}_{args.triplet_method["name"]}_node_attr.pickle')
        edge_attr = read_pickle_file(processed_data_path,
                                     f'{args.dataset}_{args.triplet_method["name"]}_edge_attr.pickle')

    else:
        if triplet_method == 'LLM':
            llm_response = read_pickle_file(processed_data_path, f'{dataset_name}_LLM_response.pickle')

            # Reformat
            for idx, triplet in enumerate(llm_response):
                formatted_triplet = triplet

                for i, item in enumerate(triplet):
                    formatted_item = convert_to_uppercase(item)
                    formatted_item = formatted_item.replace('"', '')
                    formatted_triplet[i] = formatted_item

                llm_response[idx] = formatted_triplet

            # Remove Duplicate Triplets
            llm_response = np.array(llm_response)
            llm_response = np.unique(llm_response, axis=0)

            # Remove Impossible Nodes
            src_dst = llm_response[:, [0, -1]]
            stored_nodes = list(node_name_to_id.keys())
            mask = np.ones(len(llm_response), dtype=bool)
            indices_to_remove = []

            for idx, pair in enumerate(src_dst):
                for node in pair:
                    if node not in stored_nodes:
                        indices_to_remove.append(idx)

            mask[indices_to_remove] = False
            llm_response = llm_response[mask]

            selection = np.random.choice(llm_response.shape[0], triplet_args['triplet_num'], replace=False)
            llm_triplets = llm_response[selection]

            node_names = list(node_name_to_id.keys())
            node_names = np.array(node_names)
            edge_names = llm_triplets[:, 1]
            # unique_edges, rev_map = np.unique(edge_names, return_inverse=True)

            # Get Embedding from BERT
            llm_node_embeddings = get_BERT_embeddings(node_names)
            llm_edge_embeddings = get_BERT_embeddings(edge_names)
            '''
            # Clustering
            kmeans = KMeans(n_clusters=triplet_args['cluster_num'], random_state=0)
            kmeans.fit(llm_embeddings)
            labels = kmeans.labels_

            # Assign Cluster Representatives
            for cluster_idx in range(triplet_args['cluster_num']):
                mask = torch.zeros(len(labels), dtype=bool)
                idx_loc = labels == cluster_idx
                mask[idx_loc] = True

                cluster_edge_names = unique_edges[mask]
                cluster_rep_name = cluster_edge_names[0]
                unique_edges[mask] = cluster_rep_name

                cluster_edge_attr = llm_embeddings[mask]
                cluster_rep_attr = torch.mean(cluster_edge_attr, dim=0)
                llm_embeddings[mask] = cluster_rep_attr

            full_edges = unique_edges[rev_map]
            full_edge_attr = llm_embeddings[rev_map]
            llm_response[:, 1] = full_edges
            llm_triplets, index = np.unique(llm_response, axis=0, return_index=True)
            '''

            triplet_id_to_info = {}
            for idx in range(len(llm_triplets)):
                triplet_id_to_info[idx + 1] = llm_triplets[idx].tolist()

            # edge_attr = full_edge_attr[index].numpy()
            node_attr = llm_node_embeddings.numpy()
            edge_attr = llm_edge_embeddings.numpy()

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
            freq /= freq.max()

            node_names = np.array(list(node_name_to_id.keys()))
            src = node_names[co_occurrence_idx[0]]
            dst = node_names[co_occurrence_idx[1]]

            triplet_id_to_info = {}
            for idx in range(len(src)):
                triplet_id_to_info[idx + 1] = [src[idx], freq[idx], dst[idx]]

            node_attr = None
            edge_attr = freq

        elif triplet_method == 'random':
            triplet_num = triplet_args['triplet_num']

            node_names = list(node_name_to_id.keys())
            x = np.array(node_names)
            edge_pairs = list(itertools.permutations(node_names, 2))
            edge_pairs = np.array(edge_pairs)

            selection = np.random.choice(edge_pairs.shape[0], triplet_num, replace=False)
            selected_pairs = edge_pairs[selection]

            triplet_id_to_info = {}
            for idx, pair in enumerate(selected_pairs):
                triplet_id_to_info[idx + 1] = [pair[0], 1, pair[1]]

            node_attr = None
            edge_attr = np.expand_dims(np.ones(len(selected_pairs)), axis=1)

        if save_triplets:
            save_with_pickle(triplet_id_to_info, processed_data_path,
                             f'{dataset_name}_{triplet_method}_triplet_id_to_info.pickle')
            save_with_pickle(node_attr, processed_data_path, f'{dataset_name}_{triplet_method}_node_attr.pickle')
            save_with_pickle(edge_attr, processed_data_path, f'{dataset_name}_{triplet_method}_edge_attr.pickle')

    assert len(triplet_id_to_info) == len(edge_attr), 'Number of triplets and number of edge_attr must match'

    return triplet_id_to_info, node_attr, edge_attr


def construct_graph(node_name_to_id: dict,
                    triplet_id_to_info: dict,
                    node_attr: np.ndarray,
                    edge_attr: np.ndarray,
                    processed_data_path: str,
                    dataset_name: str,
                    triplet_method: str,
                    PAD_ID: int = 0,
                    save_graph: bool = False,
                    save_triplet_maps: bool = False):

    if osp.exists(osp.join(
            processed_data_path, f'{dataset_name}_{triplet_method}_triplet_id_to_edge_index')) and osp.exists(
                osp.join(processed_data_path, f'{dataset_name}_{triplet_method}_triplet_edge_index_to_id')):
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
        edge_index = torch.tensor(edge_index)

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

    if osp.exists(osp.join(processed_data_path, f'{dataset_name}_{triplet_method}_graph.pickle')):
        graph = read_pickle_file(processed_data_path, f'{dataset_name}_{triplet_method}_graph.pickle')

    else:
        node_attr = torch.from_numpy(node_attr)

        edge_index = list(triplet_id_to_edge_index.values())
        edge_index = torch.tensor(edge_index, dtype=int).t()

        if triplet_method == 'co-occurrence':
            edge_attr = torch.from_numpy(edge_attr).unsqueeze(-1)
        elif triplet_method == 'LLM' or triplet_method == 'random':
            edge_attr = torch.from_numpy(edge_attr)

        edge_pairs = edge_index.t()
        edge_ids = []
        for pair in edge_pairs:
            edge_ids.append(triplet_edge_index_to_id[str(pair.tolist())])
        edge_ids = torch.tensor(edge_ids, dtype=int)

        pad_node_attr = torch.randn((1, node_attr.size(-1)))
        node_attr = torch.cat((pad_node_attr, node_attr), dim=0)
        pad_edge_index = torch.tensor([[PAD_ID], [PAD_ID]], dtype=int)
        edge_index = torch.cat((pad_edge_index, edge_index), dim=1)
        pad_edge_attr = torch.randn((1, edge_attr.size(-1)))
        edge_attr = torch.cat((pad_edge_attr, edge_attr), dim=0)
        pad_edge_ids = torch.tensor([0], dtype=int)
        edge_ids = torch.cat((pad_edge_ids, edge_ids), dim=0)

        graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, edge_ids=edge_ids)

        if save_graph:
            save_with_pickle(graph, processed_data_path, f'{dataset_name}_{triplet_method}_graph.pickle')

    return graph, triplet_id_to_edge_index, triplet_edge_index_to_id


def construct_patient(patient,
                      task: str,
                      dataset_name: str,
                      patient_data: dict,
                      graph: Data,
                      visit_id_to_nodes: dict,
                      node_id_to_name: dict,
                      node_name_to_id: dict,
                      triplet_edge_index_to_id: dict,
                      node_maps: tuple,
                      visit_pad_dim: int,
                      prescriptions_code_to_name: dict = None):

    patient_data['patient_id'] = patient.patient_id
    diagnoses_names = list(node_maps[0].keys())
    procedures_names = list(node_maps[1].keys())
    prescriptions_names = list(node_maps[2].keys())
    subset_threshold = torch.max(graph.edge_index)

    patient_stop_idx = len(patient)
    if task == 'drug_recommendation':
        if dataset_name == 'mimiciv':
            key = 'prescriptions'
        elif dataset_name == 'mimiciii':
            key = 'PRESCRIPTIONS'

        last_visit_drugs = patient[len(patient) - 1].get_code_list(key)

        if len(last_visit_drugs) == 0:
            second_last_visit_drugs = patient[len(patient) - 2].get_code_list(key)

            if len(second_last_visit_drugs) == 0:
                return
            else:
                patient_stop_idx = len(patient) - 1

    visit_encounters = []
    visit_discharges = []

    for idx in range(patient_stop_idx):
        visit = patient[idx]

        codes = visit_id_to_nodes[visit.visit_id]
        patient_data['visit_ids'].append(visit.visit_id)
        visit_encounters.append(visit.encounter_time)
        visit_discharges.append(visit.discharge_time)

        if task == 'drug_recommendation' and idx == patient_stop_idx - 1:
            last_visit_drugs = visit.get_code_list(key)

            last_visit_drug_names = []
            for code in last_visit_drugs:
                name = prescriptions_code_to_name[code]
                last_visit_drug_names.append(name)
                codes.remove(name)

            patient_data['last_visit_drug_names'] = last_visit_drug_names

        single_visit_node_ids = []
        single_visit_edge_ids = []
        single_visit_mask = []
        single_visit_node_type = []

        for code in codes:
            single_visit_node_ids.append(node_name_to_id[code])

        subset = torch.tensor(single_visit_node_ids, dtype=int)
        subset = subset[subset <= subset_threshold]

        visit_edge_index, visit_edge_attr = subgraph(subset=subset,
                                                     edge_index=graph.edge_index,
                                                     edge_attr=graph.edge_attr,
                                                     relabel_nodes=False)

        single_visit_node_ids = torch.unique(visit_edge_index.flatten()).tolist()
        single_visit_mask = [False] * len(single_visit_node_ids)

        if len(single_visit_node_ids) == 0:
            return

        visit_edge_attr = torch.zeros(visit_edge_index.size(-1), 1, dtype=float)

        for id in single_visit_node_ids:
            code = node_id_to_name[id]
            if code in diagnoses_names:
                single_visit_node_type.append(1)
            elif code in procedures_names:
                single_visit_node_type.append(2)
            elif code in prescriptions_names:
                single_visit_node_type.append(3)
            else:
                raise ValueError('Error code name')

        patient_data['visit_node_ids'].append(single_visit_node_ids)
        patient_data['node_padding_mask'].append(single_visit_mask)
        patient_data['visit_edge_index'].append(visit_edge_index)
        patient_data['visit_edge_attr'].append(visit_edge_attr)
        patient_data['visit_node_type'].append(single_visit_node_type)

        edge_pairs = visit_edge_index.t()
        for pair in edge_pairs:
            id = triplet_edge_index_to_id[str(pair.tolist())]
            single_visit_edge_ids.append(id)
        patient_data['visit_edge_ids'].append(single_visit_edge_ids)

    visit_node_ids = patient_data['visit_node_ids']
    visit_nodes = torch.zeros(visit_pad_dim, len(node_name_to_id) + 1)
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


def construct_pretrain_patient(patient, patient_data: dict, graph: Data, visit_id_to_nodes: dict, node_id_to_name: dict,
                               node_name_to_id: dict, triplet_edge_index_to_id: dict, node_maps: tuple,
                               visit_pad_dim: int):

    pretrain_samples = []

    patient_data['patient_id'] = patient.patient_id
    diagnoses_names = list(node_maps[0].keys())
    procedures_names = list(node_maps[1].keys())
    prescriptions_names = list(node_maps[2].keys())
    subset_threshold = torch.max(graph.edge_index)

    visit_encounters = []
    visit_discharges = []

    for visit in patient:
        sample_patient_data = deepcopy(patient_data)

        codes = visit_id_to_nodes[visit.visit_id]
        sample_patient_data['visit_ids'].append(visit.visit_id)
        visit_encounters.append(visit.encounter_time)
        visit_discharges.append(visit.discharge_time)

        single_visit_node_ids = []
        single_visit_edge_ids = []
        single_visit_mask = []
        single_visit_node_type = []

        for code in codes:
            single_visit_node_ids.append(node_name_to_id[code])

        subset = torch.tensor(single_visit_node_ids, dtype=int)
        subset = subset[subset <= subset_threshold]

        visit_edge_index, visit_edge_attr = subgraph(subset=subset,
                                                     edge_index=graph.edge_index,
                                                     edge_attr=graph.edge_attr,
                                                     relabel_nodes=False)

        single_visit_node_ids = torch.unique(visit_edge_index.flatten()).tolist()
        single_visit_mask = [False] * len(single_visit_node_ids)

        if len(single_visit_node_ids) == 0:
            return

        visit_edge_attr = torch.zeros(visit_edge_index.size(-1), 1, dtype=float)

        for id in single_visit_node_ids:
            code = node_id_to_name[id]
            if code in diagnoses_names:
                single_visit_node_type.append(1)
            elif code in procedures_names:
                single_visit_node_type.append(2)
            elif code in prescriptions_names:
                single_visit_node_type.append(3)
            else:
                raise ValueError('Error code name')

        sample_patient_data['visit_node_ids'].append(single_visit_node_ids)
        sample_patient_data['node_padding_mask'].append(single_visit_mask)
        sample_patient_data['visit_edge_index'].append(visit_edge_index)
        sample_patient_data['visit_edge_attr'].append(visit_edge_attr)
        sample_patient_data['visit_node_type'].append(single_visit_node_type)

        edge_pairs = visit_edge_index.t()
        for pair in edge_pairs:
            id = triplet_edge_index_to_id[str(pair.tolist())]
            single_visit_edge_ids.append(id)
        sample_patient_data['visit_edge_ids'].append(single_visit_edge_ids)

        pretrain_samples.append(sample_patient_data)

    visit_rel_times = []
    hist_time = 0

    for enc, dis in zip(visit_encounters, visit_discharges):
        if hist_time == 0:
            visit_rel_times.append(0)
        else:
            day_dif = (enc - hist_time).days
            week_dif = abs(day_dif) // 7
            visit_rel_times.append(week_dif + 1)

        hist_time = dis

    for idx in range(len(pretrain_samples) - 1):
        pretrain_samples[idx]['visit_rel_times'].extend(visit_rel_times[:idx + 1])

        if idx != len(pretrain_samples) - 1:
            gru_label = pretrain_samples[idx + 1]['visit_node_ids'][0]
            pretrain_samples[idx]['gru_label'][gru_label] = 1

        if idx != 0:
            pretrain_samples[idx]['visit_ids'] = pretrain_samples[idx -
                                                                  1]['visit_ids'] + pretrain_samples[idx]['visit_ids']
            pretrain_samples[idx]['visit_node_ids'] = pretrain_samples[
                idx - 1]['visit_node_ids'] + pretrain_samples[idx]['visit_node_ids']
            pretrain_samples[idx]['node_padding_mask'] = pretrain_samples[
                idx - 1]['node_padding_mask'] + pretrain_samples[idx]['node_padding_mask']
            pretrain_samples[idx]['visit_edge_index'] = pretrain_samples[
                idx - 1]['visit_edge_index'] + pretrain_samples[idx]['visit_edge_index']
            pretrain_samples[idx]['visit_edge_attr'] = pretrain_samples[
                idx - 1]['visit_edge_attr'] + pretrain_samples[idx]['visit_edge_attr']
            pretrain_samples[idx]['visit_node_type'] = pretrain_samples[
                idx - 1]['visit_node_type'] + pretrain_samples[idx]['visit_node_type']
            pretrain_samples[idx]['visit_edge_ids'] = pretrain_samples[
                idx - 1]['visit_edge_ids'] + pretrain_samples[idx]['visit_edge_ids']

    pretrain_samples = pretrain_samples[:-1]

    for sample_idx, sample in enumerate(pretrain_samples):
        for idx, single_visit_node_ids in enumerate(sample['visit_node_ids']):
            pretrain_samples[sample_idx]['set_trans_label'][idx][single_visit_node_ids] = 1

    return pretrain_samples


def organize_dataset(patients,
                     task: str,
                     graph: Data,
                     visit_id_to_nodes: list,
                     node_id_to_name: dict,
                     node_name_to_id: dict,
                     triplet_edge_index_to_id: dict,
                     node_maps: tuple,
                     visit_pad_dim: int,
                     processed_data_path: str,
                     dataset_name: str,
                     triplet_method: str,
                     save_dataset: bool = False,
                     prescriptions_code_to_name: dict = None):

    if osp.exists(osp.join(processed_data_path, f'{dataset_name}_{triplet_method}_organized_pretrain.pickle')):
        pretrain_dataset = read_pickle_file(processed_data_path,
                                            f'{dataset_name}_{triplet_method}_organized_pretrain.pickle')

    else:
        pretrain_dataset = {}

        PRETRAIN_PATIENT_TEMPLATE = {
            'patient_id': None,
            'visit_ids': [],
            'visit_node_ids': [],
            'visit_edge_ids': [],
            'visit_edge_index': [],
            'visit_edge_attr': [],
            'visit_rel_times': [],
            'visit_order': None,
            'visit_node_type': [],
            'node_padding_mask': [],
            'set_trans_label': torch.zeros(visit_pad_dim, graph.x.size(0)),
            'gru_label': torch.zeros(graph.x.size(0))
        }

        for id, patient in tqdm(patients.items(), desc='Organizing pretrain patients'):
            pretrain_patient_data = deepcopy(PRETRAIN_PATIENT_TEMPLATE)

            pretrain_patient_data = construct_pretrain_patient(patient=patient,
                                                               patient_data=pretrain_patient_data,
                                                               graph=graph,
                                                               visit_id_to_nodes=visit_id_to_nodes,
                                                               node_id_to_name=node_id_to_name,
                                                               node_name_to_id=node_name_to_id,
                                                               triplet_edge_index_to_id=triplet_edge_index_to_id,
                                                               node_maps=node_maps,
                                                               visit_pad_dim=visit_pad_dim)

            if pretrain_patient_data is not None:
                for idx, pretrain_pateint in enumerate(pretrain_patient_data):
                    pretrain_id = id + f'_{idx}'
                    pretrain_dataset[pretrain_id] = pretrain_pateint

        if save_dataset:
            save_with_pickle(pretrain_dataset, processed_data_path,
                             f'{dataset_name}_{triplet_method}_organized_pretrain.pickle')

    if task == 'drug_recommendation':
        assert prescriptions_code_to_name is not None, f'Require prescriptions_code_to_name for drug_recommendation'

    if task == 'drug_recommendation' and osp.exists(
            osp.join(processed_data_path, f'{dataset_name}_{triplet_method}_organized_DR.pickle')):
        dataset = read_pickle_file(processed_data_path, f'{dataset_name}_{triplet_method}_organized_DR.pickle')

    elif task != 'drug_recommendation' and osp.exists(
            osp.join(processed_data_path, f'{dataset_name}_{triplet_method}_organized.pickle')):
        dataset = read_pickle_file(processed_data_path, f'{dataset_name}_{triplet_method}_organized.pickle')

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
            'visit_node_type': [],
            'ehr_nodes': None,
            'node_padding_mask': [],
            'last_visit_drug_names': None,
            'label': None
        }

        for id, patient in tqdm(patients.items(), desc='Organizing patients'):
            patient_data = deepcopy(PATIENT_TEMPLATE)

            patient_data = construct_patient(patient=patient,
                                             task=task,
                                             dataset_name=dataset_name,
                                             patient_data=patient_data,
                                             graph=graph,
                                             visit_id_to_nodes=visit_id_to_nodes,
                                             node_id_to_name=node_id_to_name,
                                             node_name_to_id=node_name_to_id,
                                             triplet_edge_index_to_id=triplet_edge_index_to_id,
                                             node_maps=node_maps,
                                             visit_pad_dim=visit_pad_dim,
                                             prescriptions_code_to_name=prescriptions_code_to_name)

            if patient_data is not None:
                dataset[id] = patient_data

        if task == 'drug_recommendation' and save_dataset:
            save_with_pickle(dataset, processed_data_path, f'{dataset_name}_{triplet_method}_organized_DR.pickle')

        elif task != 'drug_recommendation' and save_dataset:
            save_with_pickle(dataset, processed_data_path, f'{dataset_name}_{triplet_method}_organized.pickle')

    return dataset, pretrain_dataset


def pad_dataset(dataset: dict,
                task: str,
                code_pad_dim: int,
                visit_pad_dim: int,
                processed_data_path: str,
                dataset_name: str,
                triplet_method: str,
                PAD_ID: int = 0,
                save_dataset: bool = False):

    if task == 'drug_recommendation' and osp.exists(
            osp.join(processed_data_path, f'{dataset_name}_{triplet_method}_padded_DR.pickle')):
        dataset = read_pickle_file(processed_data_path, f'{dataset_name}_{triplet_method}_padded_DR.pickle')

    elif task != 'drug_recommendation' and osp.exists(
            osp.join(processed_data_path, f'{dataset_name}_{triplet_method}_padded.pickle')):
        dataset = read_pickle_file(processed_data_path, f'{dataset_name}_{triplet_method}_padded.pickle')

    else:
        for patient in tqdm(dataset.values(), desc='Padding'):

            visit_node_ids = deepcopy(patient['visit_node_ids'])
            visit_edge_ids = deepcopy(patient['visit_edge_ids'])
            node_padding_mask = deepcopy(patient['node_padding_mask'])
            visit_edge_index = deepcopy(patient['visit_edge_index'])
            visit_edge_attr = deepcopy(patient['visit_edge_attr'])
            visit_rel_times = deepcopy(patient['visit_rel_times'])
            visit_nodes = deepcopy(patient['visit_nodes'])
            visit_node_type = deepcopy(patient['visit_node_type'])
            ehr_nodes = deepcopy(patient['ehr_nodes'])

            # code padding and mask padding
            for idx, single_visit_ids in enumerate(visit_node_ids):
                if len(single_visit_ids) < code_pad_dim:
                    pad_length = code_pad_dim - len(single_visit_ids)
                    visit_node_ids[idx] = single_visit_ids + [PAD_ID] * pad_length
                    visit_node_type[idx] += [0] * pad_length
                    node_padding_mask[idx] += [True] * pad_length

                elif len(single_visit_ids) > code_pad_dim:
                    raise ValueError('code_pad_dim must exceed max code length')

            if len(visit_node_ids) < visit_pad_dim:
                pad_length = visit_pad_dim - len(visit_node_ids)

                for _ in range(pad_length):
                    visit_node_ids.append([PAD_ID] * code_pad_dim)
                    visit_node_type.append([0] * code_pad_dim)
                    node_padding_mask.append([True] * code_pad_dim)

            elif len(visit_node_ids) > visit_pad_dim:
                raise ValueError('visit_pad_dim must exceed max visit num')

            visit_node_ids = torch.tensor(visit_node_ids,
                                          dtype=int) if not isinstance(visit_node_ids, torch.Tensor) else visit_node_ids

            for idx, (single_visit_edge_index,
                      single_visit_edge_attr) in enumerate(zip(visit_edge_index, visit_edge_attr)):
                if torch.eq(visit_node_ids[idx], PAD_ID).any():
                    visit_edge_ids[idx].append(PAD_ID)
                    visit_edge_index[idx] = torch.cat((torch.tensor([[PAD_ID], [PAD_ID]]), single_visit_edge_index),
                                                      dim=-1)
                    pad_length = single_visit_edge_attr.size(-1)
                    visit_edge_attr[idx] = torch.cat(
                        (torch.tensor([[PAD_ID] * pad_length], dtype=float), single_visit_edge_attr), dim=0)
                visit_edge_ids[idx] = torch.tensor(visit_edge_ids[idx], dtype=int)

            visit_rel_times = torch.tensor(
                visit_rel_times, dtype=int) if not isinstance(visit_rel_times, torch.Tensor) else visit_rel_times
            pad = torch.zeros(visit_pad_dim - len(visit_rel_times))
            visit_rel_times = torch.cat((visit_rel_times, pad), dim=0).to(torch.int)

            visit_order = torch.arange(visit_pad_dim, dtype=int)
            visit_nodes = torch.tensor(visit_nodes,
                                       dtype=int) if not isinstance(visit_nodes, torch.Tensor) else visit_nodes
            visit_node_type = torch.tensor(
                visit_node_type, dtype=int) if not isinstance(visit_node_type, torch.Tensor) else visit_node_type
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
            patient['visit_node_type'] = visit_node_type
            patient['ehr_nodes'] = ehr_nodes
            patient['node_padding_mask'] = node_padding_mask

        if task == 'drug_recommendation' and save_dataset:
            save_with_pickle(dataset, processed_data_path, f'{dataset_name}_{triplet_method}_padded_DR.pickle')

        elif task != 'drug_recommendation' and save_dataset:
            save_with_pickle(dataset, processed_data_path, f'{dataset_name}_{triplet_method}_padded.pickle')

    return dataset


def pad_pretrain_dataset(dataset: dict,
                         code_pad_dim: int,
                         visit_pad_dim: int,
                         processed_data_path: str,
                         dataset_name: str,
                         triplet_method: str,
                         PAD_ID: int = 0,
                         save_dataset: bool = False):

    if osp.exists(osp.join(processed_data_path, f'{dataset_name}_{triplet_method}_padded_pretrain.pickle')):
        dataset = read_pickle_file(processed_data_path, f'{dataset_name}_{triplet_method}_padded_pretrain.pickle')

    else:
        for patient in tqdm(dataset.values(), desc='Pretrain Padding'):
            visit_node_ids = deepcopy(patient['visit_node_ids'])
            visit_edge_ids = deepcopy(patient['visit_edge_ids'])
            node_padding_mask = deepcopy(patient['node_padding_mask'])
            visit_edge_index = deepcopy(patient['visit_edge_index'])
            visit_edge_attr = deepcopy(patient['visit_edge_attr'])
            visit_rel_times = deepcopy(patient['visit_rel_times'])
            visit_node_type = deepcopy(patient['visit_node_type'])

            # code padding and mask padding
            for idx, single_visit_ids in enumerate(visit_node_ids):
                if len(single_visit_ids) < code_pad_dim:
                    pad_length = code_pad_dim - len(single_visit_ids)
                    visit_node_ids[idx] = single_visit_ids + [PAD_ID] * pad_length
                    visit_node_type[idx] += [0] * pad_length
                    node_padding_mask[idx] += [True] * pad_length

                elif len(single_visit_ids) > code_pad_dim:
                    raise ValueError('code_pad_dim must exceed max code length')

            if len(visit_node_ids) < visit_pad_dim:
                pad_length = visit_pad_dim - len(visit_node_ids)

                for _ in range(pad_length):
                    visit_node_ids.append([PAD_ID] * code_pad_dim)
                    visit_node_type.append([0] * code_pad_dim)
                    node_padding_mask.append([True] * code_pad_dim)

            elif len(visit_node_ids) > visit_pad_dim:
                raise ValueError('visit_pad_dim must exceed max visit num')

            visit_node_ids = torch.tensor(visit_node_ids,
                                          dtype=int) if not isinstance(visit_node_ids, torch.Tensor) else visit_node_ids

            for idx, (single_visit_edge_index,
                      single_visit_edge_attr) in enumerate(zip(visit_edge_index, visit_edge_attr)):
                if torch.eq(visit_node_ids[idx], PAD_ID).any():
                    visit_edge_ids[idx].append(PAD_ID)
                    visit_edge_index[idx] = torch.cat((torch.tensor([[PAD_ID], [PAD_ID]]), single_visit_edge_index),
                                                      dim=-1)
                    pad_length = single_visit_edge_attr.size(-1)
                    visit_edge_attr[idx] = torch.cat(
                        (torch.tensor([[PAD_ID] * pad_length], dtype=float), single_visit_edge_attr), dim=0)
                visit_edge_ids[idx] = torch.tensor(visit_edge_ids[idx], dtype=int)

            visit_rel_times = torch.tensor(
                visit_rel_times, dtype=int) if not isinstance(visit_rel_times, torch.Tensor) else visit_rel_times

            pad = torch.zeros(visit_pad_dim - len(visit_rel_times))
            visit_rel_times = torch.cat((visit_rel_times, pad), dim=0).to(torch.int)

            visit_order = torch.arange(visit_pad_dim, dtype=int)
            visit_node_type = torch.tensor(
                visit_node_type, dtype=int) if not isinstance(visit_node_type, torch.Tensor) else visit_node_type
            node_padding_mask = torch.tensor(
                node_padding_mask, dtype=bool) if not isinstance(node_padding_mask, torch.Tensor) else node_padding_mask

            patient['visit_node_ids'] = visit_node_ids
            patient['visit_edge_ids'] = visit_edge_ids
            patient['visit_edge_index'] = visit_edge_index
            patient['visit_edge_attr'] = visit_edge_attr
            patient['visit_rel_times'] = visit_rel_times
            patient['visit_order'] = visit_order
            patient['visit_node_type'] = visit_node_type
            patient['node_padding_mask'] = node_padding_mask

        if save_dataset:
            save_with_pickle(dataset, processed_data_path, f'{dataset_name}_{triplet_method}_padded_pretrain.pickle')

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
                          save_dataset=True)
    print('Got dataset')

    filtered_patients = filter_dataset(patients=dataset.patients,
                                       patient_info=patient_info,
                                       filter_args=args.dataset_filtering['args'],
                                       processed_data_path=processed_data_path,
                                       dataset_name=args.dataset,
                                       save_dataset=True)
    print('Filtered dataset')

    triplet_id_to_info, node_attr, edge_attr = get_triplets(triplet_method=args.triplet_method['name'],
                                                            triplet_args=args.triplet_method['args'],
                                                            node_name_to_id=file_lib['node_name_to_id'],
                                                            node_maps=(diagnoses_code_to_name, procedures_code_to_name,
                                                                       prescriptions_code_to_name),
                                                            patients=filtered_patients,
                                                            processed_data_path=processed_data_path,
                                                            dataset_name=args.dataset,
                                                            save_triplets=True)
    print('Got triplets')

    graph, triplet_id_to_edge_index, triplet_edge_index_to_id = construct_graph(
        node_name_to_id=file_lib['node_name_to_id'],
        triplet_id_to_info=triplet_id_to_info,
        node_attr=node_attr,
        edge_attr=edge_attr,
        processed_data_path=processed_data_path,
        dataset_name=args.dataset,
        triplet_method=args.triplet_method['name'],
        save_graph=True,
        save_triplet_maps=True)
    print('Graph constructed')

    dataset, pretrain_dataset = organize_dataset(patients=filtered_patients,
                                                 task=args.task,
                                                 graph=graph,
                                                 visit_id_to_nodes=file_lib['visit_id_to_nodes'],
                                                 node_id_to_name=file_lib['node_id_to_name'],
                                                 node_name_to_id=file_lib['node_name_to_id'],
                                                 triplet_edge_index_to_id=triplet_edge_index_to_id,
                                                 node_maps=(diagnoses_name_to_code, procedures_name_to_code,
                                                            prescriptions_name_to_code),
                                                 visit_pad_dim=args.dataset_filtering['args']['visit_thresh'],
                                                 processed_data_path=processed_data_path,
                                                 dataset_name=args.dataset,
                                                 triplet_method=args.triplet_method['name'],
                                                 save_dataset=True,
                                                 prescriptions_code_to_name=prescriptions_code_to_name)
    print('Organized dataset')

    padded_dataset = pad_dataset(dataset=dataset,
                                 task=args.task,
                                 code_pad_dim=args.dataset_filtering['args']['code_thresh'],
                                 visit_pad_dim=args.dataset_filtering['args']['visit_thresh'],
                                 processed_data_path=processed_data_path,
                                 dataset_name=args.dataset,
                                 triplet_method=args.triplet_method['name'],
                                 save_dataset=True)

    padded_pretrain_dataset = pad_pretrain_dataset(dataset=pretrain_dataset,
                                                   code_pad_dim=args.dataset_filtering['args']['code_thresh'],
                                                   visit_pad_dim=args.dataset_filtering['args']['visit_thresh'],
                                                   processed_data_path=processed_data_path,
                                                   dataset_name=args.dataset,
                                                   triplet_method=args.triplet_method['name'],
                                                   save_dataset=True)

    print('Padded dataset')
    print('Data preparation complete')


if __name__ == '__main__':
    args = get_args()
    run(args=args)
