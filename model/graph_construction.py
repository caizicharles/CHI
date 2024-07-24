import numpy as np
import re
import itertools
from typing import List, Dict
import pyhealth.medcode as pymed
import torch
from torch_geometric.data import HeteroData
from openai import OpenAI
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.utils import *


def generate_graph(nodes: list, triplets: list, triplet_method: str):

    graph = HeteroData()

    nodes = np.array(nodes)
    for node_idx, node_type in nodes:
        graph[node_type].num_nodes = 1

    for triplet in triplets:
        full_edge_type = triplet[1:]
        src_node_idx = np.where(nodes == triplet[1])[0][0]
        dst_node_idx = np.where(nodes == triplet[3])[0][0]

        if triplet_method == 'co-occurrence':
            full_edge_type[1] = 'to'
            full_edge_type = tuple(full_edge_type)
            graph[full_edge_type].edge_weight = triplet[2]

        else:
            full_edge_type = tuple(full_edge_type)

        graph[full_edge_type].edge_index = torch.tensor([[src_node_idx], [dst_node_idx]], dtype=torch.long)

    return graph


def get_triplets(triplet_method: str,
                 LLM_answers: list = None,
                 LLM_prompts: list = None,
                 nodes: list = None,
                 node_maps: tuple = None,
                 patients: dict = None,
                 co_threshold: int = 0):

    assert triplet_method in ['LLM', 'MedBERT', 'co-occurrence'], \
    f'Unsupported triplet_mthod: {triplet_method}'

    if triplet_method == 'LLM':
        if LLM_answers is not None:
            return LLM_answers_to_triplets(LLM_answers)
        else:
            assert LLM_prompts is not None, 'Need LLM_prompts to compute triplets using GPT.'
            # TODO

    elif triplet_method == 'MedBERT':
        # TODO
        pass

    elif triplet_method == 'co-occurrence':
        assert nodes is not None, 'Need nodes to calculate co-occurence.'
        assert node_maps is not None, 'Need node map to calculate co-occurence.'
        assert patients is not None, 'Need dataset to calculate co-occurence.'

        nodes = np.array(nodes)
        co_occurrence_freq = np.zeros((len(nodes), len(nodes)))

        for patient in tqdm(patients.values()):
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
                    occurred_codes[idx] = np.where(nodes == code)[0][0]

                if len(occurred_codes) > 1:
                    pairwise_combinations = list(itertools.permutations(occurred_codes, 2))
                    rows, cols = zip(*pairwise_combinations)
                    co_occurrence_freq[np.array(rows), np.array(cols)] += 1

        co_occurrence_idx = np.where(co_occurrence_freq >= co_threshold)
        freq = co_occurrence_freq[co_occurrence_idx[0], co_occurrence_idx[1]]
        
        '''
        # Row Normalization
        for row in co_occurrence_freq:
            row /= np.max(row)

        plt.imshow(co_occurrence_freq, cmap='viridis')
        plt.colorbar()
        plt.xticks(ticks=np.arange(len(nodes)), labels=nodes, rotation=90)
        plt.yticks(ticks=np.arange(len(nodes)), labels=nodes)

        plt.xlabel('Columns')
        plt.ylabel('Rows')

        plt.show()
        exit()
        '''

        src = nodes[co_occurrence_idx[0], 1]
        dst = nodes[co_occurrence_idx[1], 1]

        triplets = []
        for idx in range(len(src)):
            triplets.append([idx, src[idx], freq[idx], dst[idx]])

        return triplets


def format_code_map(input: List[Dict]) -> Dict:

    code_to_name = {}
    name_to_code = {}
    for row in input:
        code_to_name |= {row['code']: row['code_name']}
        name_to_code |= {row['code_name']: row['code']}

    return code_to_name, name_to_code


def map_to_nodes(map: list):

    nodes = []
    for row in map:
        nodes.append(row['code_name'])

    return nodes


def LLM_answers_to_triplets(prompts: list):

    triplets = []
    for row in prompts:
        entry = re.findall(r'\{.*?\}', row)
        data = []
        for element in entry:
            _element = element.replace('{', '')
            _element = _element.replace('}', '')
            data.append(_element)
        triplets.append(data)

    return triplets


def get_graph_nodes(dataset):

    diagnosis_nodes = {}
    procedure_nodes = {}
    prescription_nodes = {}
    diagnosis_list = []
    procedure_list = []
    prescription_list = []

    for patient in dataset.patients.values():
        for visit in patient:
            diagnosis_list.append(visit.get_code_list('diagnoses_icd'))
            procedure_list.append(visit.get_code_list('procedures_icd'))
            prescription_list.append(visit.get_code_list('prescriptions'))

    diagnosis_flat = [d for sub in diagnosis_list for d in sub]
    procedure_flat = [pro for sub in procedure_list for pro in sub]
    prescription_flat = [pre for sub in prescription_list for pre in sub]

    diagnosis_flat = list(dict.fromkeys(diagnosis_flat))
    procedure_flat = list(dict.fromkeys(procedure_flat))
    prescription_flat = list(dict.fromkeys(prescription_flat))

    if dataset.dataset_name == 'MIMIC4Dataset':
        diag_map = pymed.InnerMap.load('CCSCM')
        proc_map = pymed.InnerMap.load('CCSPROC')
        pres_map = pymed.InnerMap.load('ATC')

    for diagnosis in diagnosis_flat:
        diagnosis_nodes[diagnosis] = diag_map.lookup(diagnosis)
    for procedure in procedure_flat:
        procedure_nodes[procedure] = proc_map.lookup(procedure)
    for prescription in prescription_flat:
        if prescription == '0':
            prescription_nodes[prescription] = pres_map.lookup(prescription)  #'sodium chloride'
        else:
            prescription_nodes[prescription] = pres_map.lookup(prescription)

    return diagnosis_nodes, procedure_nodes, prescription_nodes
