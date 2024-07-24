import random
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import tensorboardX

import matplotlib.pyplot as plt
from prettytable import PrettyTable

# from model import get_graph_nodes, map_to_nodes, generate_graph

from utils.misc import init_logger
from utils.args import get_args
from utils.utils import *
from utils.optimizer import OPTIMIZERS
from utils.scheduler import SCHEDULERS

from dataset.data_processing import preprocess_dataset
from dataset.dataset import MIMICIVBaseDataset, split_by_patient
from dataloader.dataloader import DataLoader

from model.graph_construction import get_triplets, format_code_map
from model.model import MODELS
from model.criterion import CRITERIONS

from metrics.metrics import METRICS

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


def task_configuring_model(args, prescriptions):

    task = args.task

    if task == 'mortality_prediction' or task == 'readmission_prediction':
        out_dim = 1

    elif task == 'drug_recommendation':
        out_dim = len(prescriptions)

    elif task == 'los_prediction':
        out_dim = 10

    return out_dim


def main(args):

    writer = init_logger(args)
    logger.info('Process begins...')
    logger.info(f'Task: {args.task}')

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

    if triplets is not None:
        triplets = get_triplets(triplet_method=args.triplet_method,
                                nodes=nodes,
                                node_maps=(diagnoses_maps[0], procedures_maps[0], prescriptions_maps[0]),
                                patients=patients,
                                co_threshold=3)

    # graph = generate_graph(nodes, triplets, args.triplet_method)
    # logger.info('Graph is generated')

    train_dataset = MIMICIVBaseDataset(patients=train_patients,
                                       task=args.task,
                                       nodes=nodes,
                                       triplets=triplets,
                                       diagnoses_map=diagnoses_maps,
                                       procedures_map=procedures_maps,
                                       prescriptions_map=prescriptions_maps,
                                       nodes_in_visits=nodes_in_visits,
                                       triplet_method=args.triplet_method,
                                       code_pad_dim=args.pad_dim,
                                       visit_pad_dim=args.visit_thresh)
    exit()

    val_dataset = MIMICIVBaseDataset(patients=val_patients,
                                     task=args.task,
                                     nodes=nodes,
                                     triplets=triplets,
                                     diagnoses_map=diagnoses_maps,
                                     procedures_map=procedures_maps,
                                     prescriptions_map=prescriptions_maps,
                                     nodes_in_visits=nodes_in_visits,
                                     triplet_method=args.triplet_method,
                                     code_pad_dim=args.pad_dim,
                                     visit_pad_dim=args.visit_thresh)

    test_dataset = MIMICIVBaseDataset(patients=test_patients,
                                      task=args.task,
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

    train_loader = DataLoader(dataset=train_dataset, task=args.task, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, task=args.task, batch_size=args.val_batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, task=args.task, batch_size=args.test_batch_size, shuffle=False)
    logger.info('DataLoader ready')

    out_dim = task_configuring_model(args, prescriptions_maps[0])

    model_configs = args.model['args'] | {
        'device': device,
        'num_nodes': len(nodes),
        'num_edges': len(triplets),
        'visit_thresh': args.visit_thresh,
        'visit_code_num': args.code_freq_filter,
        'out_dim': out_dim
    }
    model = MODELS[args.model['name']](model_configs)
    model.to(device)
    logger.info(f'Model: {args.model["name"]} ready')

    optimizer = OPTIMIZERS[args.optimizer['name']](model.parameters(), **args.optimizer['args'])

    scheduler = None
    if args.scheduler is not None:
        scheduler = SCHEDULERS[args.scheduler['name']](optimizer, **args.scheduler['args'])

    global_iter_idx = [0]

    # count0 = 0
    # count1 = 0
    # for data in train_loader:
    #     labels = data.labels
    #     labels = labels.squeeze(-1)
    #     for l in labels:
    #         if l == 0:
    #             count0 += 1
    #         elif l == 1:
    #             count1 += 1
    # print(count0, count1, count0/(count0+count1))
    # exit()

    for epoch_idx in range(args.num_epochs):
        # Train
        train_score = single_train(
            model,
            args.model['model_type'],
            args.task,
            train_loader,
            epoch_idx,
            global_iter_idx,
            optimizer,
            criterions=[CRITERIONS[criterion](**args.criterion[criterion]) for criterion in args.criterion],
            metrics=[METRICS[metric](args.task) for metric in args.metrics],
            scheduler=scheduler,
            logging_freq=args.logging_freq,
            writer=writer)

        # Validate
        if epoch_idx % args.val_freq == 0:
            val_score = single_validate(model,
                                        args.task,
                                        val_loader,
                                        global_iter_idx[0],
                                        metrics=[METRICS[metric](args.task) for metric in args.metrics],
                                        writer=writer)


def single_train(model,
                 model_type,
                 task,
                 dataloader,
                 epoch_idx,
                 global_iter_idx,
                 optimizer,
                 criterions=[],
                 metrics=[],
                 scheduler=None,
                 logging_freq=10,
                 writer=None):

    model.train()
    epoch_loss = []
    prob_all = []
    target_all = []

    for idx, data in enumerate(dataloader):
        data = data.to(device)

        optimizer.zero_grad()

        out, prototypes = model(data.node_ids, data.edge_index, data.edge_attr, data.visit_rel_times, data.visit_order,
                                data.attn_mask)

        labels = data.labels

        loss = 0.
        if model_type == 'base':
            for criterion in criterions:
                if criterion.NAME == 'contrastive_loss':
                    loss += criterion(prototypes)
                else:
                    loss += criterion(out, labels)
        else:
            for criterion in criterions:
                loss += criterion(out, labels)

        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if task == 'los_prediction':
            probability = F.softmax(out, dim=-1)
        else:
            probability = torch.sigmoid(out)

        prob_all.append(probability.cpu().detach())
        target_all.append(labels.cpu().detach())

        if idx % logging_freq == 0:
            logger.info(
                f"Epoch: {epoch_idx:4d}, Iteration: {idx:4d} / {len(dataloader):4d} [{global_iter_idx[0]:5d}], Loss: {loss.item()}"
            )
        if writer is not None:
            writer.add_scalar('train/batch_loss', loss.item(), global_iter_idx[0])

        global_iter_idx[0] += 1

    if scheduler is not None:
        scheduler.step()

    epoch_loss_avg = np.mean(epoch_loss)
    logger.info(f"Epoch: {epoch_idx:4d},  [{global_iter_idx[0]:5d}], Epoch Loss: {epoch_loss_avg}")
    if writer is not None:
        writer.add_scalar('train/epoch_loss', epoch_loss_avg, epoch_idx)

    prob_all = np.concatenate(prob_all, axis=0)  # (N_patients,L)
    target_all = np.concatenate(target_all, axis=0)

    for metric in metrics:
        score = metric.calculate(prob_all, target_all)
        metric.log(score, logger, writer=writer, global_iter=global_iter_idx[0], name_prefix='train/')

    return


def single_validate(model, task, dataloader, global_iter_idx, metrics=[], writer=None):

    model.eval()
    prob_all = []
    target_all = []

    for _, data in enumerate(tqdm(dataloader)):
        data = data.to(device)

        with torch.no_grad():
            out, _ = model(data.node_ids, data.edge_index, data.edge_attr, data.visit_rel_times, data.visit_order,
                           data.attn_mask)

            if task == 'los_prediction':
                probability = F.softmax(out, dim=-1)
            else:
                probability = torch.sigmoid(out)

            labels = data.labels  # (B,L)

            prob_all.append(probability.cpu())
            target_all.append(labels.cpu())

    prob_all = np.concatenate(prob_all, axis=0)  # (N_patients,L)
    target_all = np.concatenate(target_all, axis=0)

    for metric in metrics:
        score = metric.calculate(prob_all, target_all)
        metric.log(score, logger, writer=writer, global_iter=global_iter_idx, name_prefix='val/')

    return


def test(model, task, dataloader, global_iter_idx, metrics=[], writer=None):

    model.eval()
    prob_all = []
    target_all = []

    for _, data in enumerate(tqdm(dataloader)):
        data = data.to(device)

        with torch.no_grad():
            out, _ = model(data.node_ids, data.edge_index, data.edge_attr, data.visit_rel_times, data.visit_order,
                           data.attn_mask)

            if task == 'los_prediction':
                probability = F.softmax(out, dim=-1)
            else:
                probability = torch.sigmoid(out)

            labels = data.labels  # (B,L)

            prob_all.append(probability.cpu())
            target_all.append(labels.cpu())

    prob_all = np.concatenate(prob_all, axis=0)  # (N_patients,L)
    target_all = np.concatenate(target_all, axis=0)

    for metric in metrics:
        score = metric.calculate(prob_all, target_all)
        metric.log(score, logger)

    return score


if __name__ == '__main__':
    args = get_args()
    main(args=args)
