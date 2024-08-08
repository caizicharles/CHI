import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import tensorboardX

from utils.misc import init_logger
from utils.args import get_args
from utils.utils import *
from dataset.dataset import MIMICIVBaseDataset, split_by_patient
from dataloader.dataloader import DataLoader
from model.model import MODELS
from trainer.optimizer import OPTIMIZERS
from trainer.scheduler import SCHEDULERS
from trainer.criterion import CRITERIONS
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
    # torch.use_deterministic_algorithms(True)


def load_all(processed_data_path: str, dataset_name: str, triplet_method: str):

    node_id_to_name = read_pickle_file(processed_data_path, f'{dataset_name}_node_id_to_name.pickle')
    triplet_id_to_info = read_pickle_file(processed_data_path,
                                          f'{dataset_name}_{triplet_method}_triplet_id_to_info.pickle')
    diagnoses_map = read_csv_file(processed_data_path, 'diagnoses_code.csv')
    procedures_map = read_csv_file(processed_data_path, 'procedures_code.csv')
    prescriptions_map = read_csv_file(processed_data_path, 'prescriptions_code.csv')

    filtered_patients = read_pickle_file(processed_data_path, f'{dataset_name}_filtered.pickle')
    full_dataset = read_pickle_file(processed_data_path, f'{dataset_name}_padded.pickle')
    full_graph = read_pickle_file(processed_data_path, f'{dataset_name}_{triplet_method}_graph.pickle')

    return {
        'node_id_to_name': node_id_to_name,
        'triplet_id_to_info': triplet_id_to_info,
        'diagnoses_map': diagnoses_map,
        'procedures_map': procedures_map,
        'prescriptions_map': prescriptions_map,
        'filtered_patients': filtered_patients,
        'full_dataset': full_dataset,
        'full_graph': full_graph
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
    logger.info(f'Process begins...')
    logger.info(f'Task: {args.task}')
    logger.info(f'Model: {args.model["name"]}')

    seed_everything(args.seed)

    file_lib = load_all(args.processed_data_path, args.dataset, args.triplet_method)
    logger.info('Completed file loading')

    diagnoses_maps = format_code_map(file_lib['diagnoses_map'])
    procedures_maps = format_code_map(file_lib['procedures_map'])
    prescriptions_maps = format_code_map(file_lib['prescriptions_map'])

    ratio = [args.train_proportion, args.val_proportion, args.test_proportion]
    train_patients, val_patients, test_patients = split_by_patient(file_lib['full_dataset'], ratio)

    train_dataset = MIMICIVBaseDataset(patients=train_patients,
                                       filtered_patients=file_lib['filtered_patients'],
                                       task=args.task,
                                       graph=file_lib['full_graph'])
    val_dataset = MIMICIVBaseDataset(patients=val_patients,
                                     filtered_patients=file_lib['filtered_patients'],
                                     task=args.task,
                                     graph=file_lib['full_graph'])
    test_dataset = MIMICIVBaseDataset(patients=test_patients,
                                      filtered_patients=file_lib['filtered_patients'],
                                      task=args.task,
                                      graph=file_lib['full_graph'])
    logger.info('Dataset ready')

    # for patient in train_dataset:
    #     x = patient['visit_node_ids']
    #     y = patient['visit_edge_index']

    #     for idx in range(len(y)):
    #         xx = torch.unique(x[idx])
    #         yy = torch.unique(y[idx].flatten())

    #         if not torch.equal(xx, yy):
    #             print(patient['patient_id'])
    #             print(xx)
    #             print(yy)
    #             print(len(xx), len(yy))
    #             exit()

    # exit()

    train_loader = DataLoader(dataset=train_dataset,
                              graph=file_lib['full_graph'],
                              model_name=args.model['name'],
                              task=args.task,
                              batch_size=args.train_batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                            graph=file_lib['full_graph'],
                            model_name=args.model['name'],
                            task=args.task,
                            batch_size=args.val_batch_size,
                            shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                             graph=file_lib['full_graph'],
                             model_name=args.model['name'],
                             task=args.task,
                             batch_size=args.test_batch_size,
                             shuffle=False)
    logger.info('DataLoader ready')

    out_dim = task_configuring_model(args, prescriptions_maps[0])

    model_configs = args.model['args'] | {
        'device': device,
        'num_nodes': len(file_lib['node_id_to_name']),
        'num_edges': len(file_lib['triplet_id_to_info']),
        'visit_thresh': args.visit_thresh,
        'visit_code_num': args.code_thresh,
        'out_dim': out_dim
    }
    model = MODELS[args.model['name']](model_configs)
    model.to(device)
    logger.info('Model ready')

    optimizer = OPTIMIZERS[args.optimizer['name']](model.parameters(), **args.optimizer['args'])

    scheduler = None
    if args.scheduler is not None:
        scheduler = SCHEDULERS[args.scheduler['name']](optimizer, **args.scheduler['args'])

    global_iter_idx = [0]

    for epoch_idx in range(args.num_epochs):
        # Train
        _ = single_train(
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
            _ = single_validate(model,
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

        additional_data = {
            'cat_node_ids': data.cat_node_ids,
            'cat_edge_ids': data.cat_edge_ids,
            'cat_edge_index': data.cat_edge_index,
            'cat_edge_attr': data.cat_edge_attr,
            'visit_nodes': data.visit_nodes,
            'ehr_nodes': data.ehr_nodes,
            'batch': data.batch,
            'batch_patient': data.batch_patient
        }

        out, prototypes = model(node_ids=data.visit_node_ids,
                                edge_idx=data.global_edge_index,
                                edge_attr=data.global_edge_attr,
                                visit_times=data.visit_rel_times,
                                visit_order=data.visit_order,
                                attn_mask=data.attn_mask,
                                **additional_data)

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
            additional_data = {
                'cat_node_ids': data.cat_node_ids,
                'cat_edge_ids': data.cat_edge_ids,
                'cat_edge_index': data.cat_edge_index,
                'cat_edge_attr': data.cat_edge_attr,
                'visit_nodes': data.visit_nodes,
                'ehr_nodes': data.ehr_nodes,
                'batch': data.batch,
                'batch_patient': data.batch_patient
            }

            out, _ = model(node_ids=data.visit_node_ids,
                           edge_idx=data.global_edge_index,
                           edge_attr=data.global_edge_attr,
                           visit_times=data.visit_rel_times,
                           visit_order=data.visit_order,
                           attn_mask=data.attn_mask,
                           **additional_data)

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
