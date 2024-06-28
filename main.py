import random
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import tensorboardX

import matplotlib.pyplot as plt

from utils import *
from dataset import preprocess_dataset, MIMICIVBaseDataset, split_by_patient, DataLoader
from model import get_graph_nodes, get_triplets, map_to_nodes, format_code_map, generate_graph, \
    OurModel, OPTIMIZERS, SCHEDULERS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_all(raw_data_path: str, save_data_path: str, graph_construction_path: str):

    patient_info = read_csv_file(raw_data_path, 'mimiciv/2.2/hosp/patients.csv')
    diagnoses_map = read_csv_file(save_data_path, 'diagnoses_code.csv')
    procedures_map = read_csv_file(save_data_path, 'procedures_code.csv')
    prescriptions_map = read_csv_file(save_data_path, 'prescriptions_code.csv')

    nodes = read_pickle_file(graph_construction_path, 'nodes.pickle')
    try:
        triplets = read_pickle_file(graph_construction_path, 'triplets.pickle')
    except:
        triplets = None

    prompts = read_txt_file(graph_construction_path, 'prompt.txt')
    gpt_answers = read_txt_file(graph_construction_path, 'gpt_4o_response.txt')

    nodes_in_visits = read_pickle_file(graph_construction_path, 'nodes_in_visits.pickle')

    return {
        'nodes': nodes,
        'triplets': triplets,
        'patient_info': patient_info,
        'diagnoses_map': diagnoses_map,
        'procedures_map': procedures_map,
        'prescriptions_map': prescriptions_map,
        'prompts': prompts,
        'gpt_answers': gpt_answers,
        'nodes_in_visits': nodes_in_visits
    }


def task_configuring_model(task, prescriptions):

    if task == 'mortality_prediction' or task == 'readmission_prediction':
        out_dim = 1
        criterion = F.binary_cross_entropy_with_logits

    elif task == 'drug_recommendation':
        out_dim = len(prescriptions)
        criterion = F.binary_cross_entropy_with_logits

    elif task == 'los_prediction':
        out_dim = 10
        criterion = F.cross_entropy

    return out_dim, criterion


def single_validate(model, device, val_loader, criterion):

    model.eval()

    for idx, data in enumerate(val_loader):
        data = data.to(device)

        with torch.no_grad():

            out = model()


def main(args):

    writer = init_logger(args)
    logger.info('Process begins...')

    seed_everything(args.seed)

    file_lib = load_all(args.raw_data_path, args.save_data_path, args.graph_construction_path)
    logger.info('Completed file loading')

    patient_info = format_from_csv(file_lib['patient_info'])
    diagnoses_maps = format_code_map(file_lib['diagnoses_map'])
    procedures_maps = format_code_map(file_lib['procedures_map'])
    prescriptions_maps = format_code_map(file_lib['prescriptions_map'])
    nodes = file_lib['nodes']
    nodes_in_visits = file_lib['nodes_in_visits']
    triplets = file_lib['triplets']

    patients = preprocess_dataset(raw_dataset_path=args.raw_data_path,
                                  dataset_name=args.dataset,
                                  patient_info=patient_info,
                                  save_path=args.save_data_path,
                                  filtered=True,
                                  filter_params={
                                      'age_thresh_low': args.age_thresh_low,
                                      'age_thresh_high': args.age_thresh_high,
                                      'code_thresh': args.code_freq_filter,
                                      'visit_thresh': args.visit_thresh
                                  })
    logger.info('Dataset preprocessing complete')
    '''
    # count = 0
    # length = []
    # for patient in patients.values():
    #     a = True
    #     if len(patient) == 1 or len(patient) > 15:
    #         continue
    #     for visit in patient:
    #         num = 0
    #         num += len(visit.get_code_list('diagnoses_icd'))
    #         num += len(visit.get_code_list('procedures_icd'))
    #         num += len(visit.get_code_list('prescriptions'))
            
    #         if num > 60 or num == 0:
    #             a = False
        
    #     if a:
    #         count += 1

    # print(count)
    # exit()

    # x = list(dict.fromkeys(length))

    # print(np.unique(length, return_counts=True))
    # plt.hist(length, bins=len(x))
    # plt.show()
    # exit()
    '''

    ratio = [args.train_proportion, args.val_proportion, args.test_proportion]
    train_patients, val_patients, test_patients = split_by_patient(patients, ratio)

    if triplets is None:
        triplets = get_triplets(triplet_method=args.triplet_method,
                                nodes=nodes,
                                node_maps=(diagnoses_maps[0], procedures_maps[0], prescriptions_maps[0]),
                                patients=patients,
                                co_threshold=3)

    # graph = generate_graph(nodes, triplets, args.triplet_method)
    # logger.info('Graph is generated')

    train_dataset = MIMICIVBaseDataset(
        patients=train_patients,
        task=args.task,
        #    graph=graph,
        nodes=nodes,
        triplets=triplets,
        diagnoses_map=diagnoses_maps,
        procedures_map=procedures_maps,
        prescriptions_map=prescriptions_maps,
        nodes_in_visits=nodes_in_visits,
        triplet_method=args.triplet_method,
        code_pad_dim=args.pad_dim,
        visit_pad_dim=args.visit_thresh)
    logger.info('Dataset ready')
    '''
    val_dataset = MIMICIVBaseDataset(dataset=val_dataset,
                                       task=args.task,
                                    #    graph=graph,
                                       nodes=nodes,
                                       triplets=triplets,
                                       diagnoses_map=diagnoses_maps,
                                       procedures_map=procedures_maps,
                                       prescriptions_map=prescriptions_maps,
                                       nodes_in_visits=nodes_in_visits,
                                       pad_dim=args.pad_dim)
    test_dataset = MIMICIVBaseDataset(dataset=test_dataset,
                                       task=args.task,
                                    #    graph=graph,
                                       nodes=nodes,
                                       triplets=triplets,
                                       diagnoses_map=diagnoses_maps,
                                       procedures_map=procedures_maps,
                                       prescriptions_map=prescriptions_maps,
                                       nodes_in_visits=nodes_in_visits,
                                       pad_dim=args.pad_dim)
    '''

    train_loader = DataLoader(dataset=train_dataset, task=args.task, batch_size=args.batch_size, shuffle=True)

    out_dim, criterion = task_configuring_model(args.task, prescriptions_maps[0])

    model = OurModel(num_nodes=len(nodes),
                     num_edges=len(triplets),
                     embed_dim=args.embed_dim,
                     gnn_hidden_dim=args.gnn_hidden_dim,
                     trans_hidden_dim=args.trans_hidden_dim,
                     out_dim=out_dim)

    model.to(device)

    optimizer = OPTIMIZERS[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = SCHEDULERS[args.scheduler]()

    global_iter_idx = 0

    for epoch_idx in range(args.num_epochs):
        # Train
        single_train(model, train_loader, epoch_idx, global_iter_idx, criterion, optimizer, writer=writer)

        # Validate
        # if epoch_idx % args.val_freq == 0:
        #     single_validate()

    return


def single_train(model,
                 dataloader,
                 epoch_idx,
                 global_iter_idx,
                 criterion,
                 optimizer,
                 scheduler=None,
                 logging_freq=10,
                 writer=None):

    model.train()

    for idx, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.node_ids, data.edge_index, data.edge_attr)

        labels = data.labels
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        if idx % logging_freq == 0:
            logger.info(
                f"Epoch: {epoch_idx:4d}, Iteration: {idx:4d} / {len(dataloader):4d} [{global_iter_idx:5d}], Loss: {loss.item()}"
            )
        if writer is not None:
            writer.add_scalar('train/loss', loss.item(), global_iter_idx)

        global_iter_idx += 1

    if scheduler is not None:
        scheduler.step()


if __name__ == '__main__':
    args = get_args()
    main(args=args)
