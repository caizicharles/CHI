import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
import tensorboardX

from utils.misc import init_logger, save_params, save_embed, save_prototype
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


def load_all(processed_data_path: str, dataset_name: str, triplet_method: str, task: str):

    node_id_to_name = read_pickle_file(processed_data_path, f'{dataset_name}_node_id_to_name.pickle')
    triplet_id_to_info = read_pickle_file(processed_data_path,
                                          f'{dataset_name}_{triplet_method}_triplet_id_to_info.pickle')
    diagnoses_map = read_csv_file(processed_data_path, 'diagnoses_code.csv')
    procedures_map = read_csv_file(processed_data_path, 'procedures_code.csv')
    prescriptions_map = read_csv_file(processed_data_path, 'prescriptions_code.csv')

    filtered_patients = read_pickle_file(processed_data_path, f'{dataset_name}_filtered.pickle')
    full_graph = read_pickle_file(processed_data_path, f'{dataset_name}_{triplet_method}_graph.pickle')

    if task == 'pretrain':
        full_dataset = read_pickle_file(processed_data_path, f'{dataset_name}_{triplet_method}_padded_pretrain.pickle')
    elif task == 'drug_recommendation':
        full_dataset = read_pickle_file(processed_data_path, f'{dataset_name}_{triplet_method}_padded_DR.pickle')
    else:
        full_dataset = read_pickle_file(processed_data_path, f'{dataset_name}_{triplet_method}_padded.pickle')

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


def task_configuring_model(args, node_id_to_name, prescriptions):

    task = args.task

    if task == 'mortality_prediction' or task == 'readmission_prediction':
        out_dim = 1
        class_num = 2

    elif task == 'drug_recommendation':
        out_dim = len(prescriptions)
        class_num = len(prescriptions)

    elif task == 'los_prediction':
        out_dim = 10
        class_num = 10

    elif task == 'pretrain':
        out_dim = len(node_id_to_name) + 1
        class_num = len(node_id_to_name) + 1

    return out_dim, class_num


def main(args):

    writer = init_logger(args)
    logger.info(f'Process begins...')
    logger.info(f'Task: {args.task}')
    logger.info(f'Triplet method: {args.triplet_method}')
    logger.info(f'Model: {args.model["name"]}')

    seed_everything(args.seed)

    file_lib = load_all(args.processed_data_path, args.dataset, args.triplet_method, args.task)
    logger.info('Completed file loading')

    diagnoses_maps = format_code_map(file_lib['diagnoses_map'])
    procedures_maps = format_code_map(file_lib['procedures_map'])
    prescriptions_maps = format_code_map(file_lib['prescriptions_map'])

    ratio = [args.train_proportion, args.val_proportion, args.test_proportion]
    train_patients, val_patients, test_patients = split_by_patient(file_lib['full_dataset'], ratio)

    train_dataset = MIMICIVBaseDataset(patients=train_patients,
                                       filtered_patients=file_lib['filtered_patients'],
                                       task=args.task,
                                       graph=file_lib['full_graph'],
                                       prescriptions_code_to_name=prescriptions_maps[0])
    val_dataset = MIMICIVBaseDataset(patients=val_patients,
                                     filtered_patients=file_lib['filtered_patients'],
                                     task=args.task,
                                     graph=file_lib['full_graph'],
                                     prescriptions_code_to_name=prescriptions_maps[0])
    test_dataset = MIMICIVBaseDataset(patients=test_patients,
                                      filtered_patients=file_lib['filtered_patients'],
                                      task=args.task,
                                      graph=file_lib['full_graph'],
                                      prescriptions_code_to_name=prescriptions_maps[0])
    logger.info('Dataset ready')

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

    out_dim, class_num = task_configuring_model(args, file_lib['node_id_to_name'], prescriptions_maps[0])

    multi_level_embed = None
    if args.model['load_embed']:
        multi_level_embed = torch.load(args.model['load_embed'])['embed']

    model_configs = args.model['args'] | {
        'device': device,
        'num_nodes': len(file_lib['node_id_to_name']),
        'num_edges': len(file_lib['triplet_id_to_info']),
        'visit_thresh': args.visit_thresh,
        'visit_code_num': args.code_thresh,
        'out_dim': out_dim,
        'class_num': class_num,
        'global_node_attr': file_lib['full_graph'].x,
        'global_edge_attr': file_lib['full_graph'].edge_attr,
        'global_edge_index': file_lib['full_graph'].edge_index,
        'global_edge_ids': file_lib['full_graph'].edge_ids,
        'multi_level_embed': multi_level_embed
    }
    model = MODELS[args.model['name']](model_configs)
    model.to(device)

    if args.pretrained:
        logger.info(f"Load pretrained model from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)

    if args.model['freeze']:
        for param in model.gnn.parameters():
            param.requires_grad = False
        for param in model.set_transformer.parameters():
            param.requires_grad = False
        for param in model.gru.parameters():
            param.requires_grad = False

    optimizer = OPTIMIZERS[args.optimizer['name']](filter(lambda p: p.requires_grad, model.parameters()),
                                                   **args.optimizer['args'])

    scheduler = None
    if args.scheduler is not None:
        scheduler = SCHEDULERS[args.scheduler['name']](optimizer, **args.scheduler['args'])

    global_iter_idx = [0]
    start_epoch = 0

    if args.pretrained:
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict({'state': ckpt['optimizer']['state'], 'param_groups': optimizer.param_groups})
        if ckpt['scheduler'] is not None:
            scheduler.load_state_dict({'state': ckpt['scheduler']['state'], 'param_groups': scheduler.param_groups})
        if ckpt['iter'] is not None:
            global_iter_idx[0] = ckpt['iter']
        if ckpt['epoch'] is not None:
            start_epoch = ckpt['epoch'] + 1

    logger.info('Model ready')

    if args.model['mode'] == 'train':
        early_stopping_counter = 0
        best_score = 0.

        for epoch_idx in range(start_epoch, args.max_epoch):
            # Train
            embed_all, label_all, attn_mask_all = single_train(
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
            if epoch_idx % args.val_freq == 0 or epoch_idx == args.max_epoch - 1:
                results, val_embed_all, val_label_all, val_attn_mask_all = single_validate(
                    model,
                    args.model['model_type'],
                    args.task,
                    val_loader,
                    epoch_idx,
                    global_iter_idx,
                    criterions=[CRITERIONS[criterion](**args.criterion[criterion]) for criterion in args.criterion],
                    metrics=[METRICS[metric](args.task) for metric in args.metrics],
                    writer=writer)

                if args.task != 'pretrain':
                    # Early Stopping
                    score = results[args.early_stopping_indicator]
                    if score >= best_score:
                        best_model = deepcopy(model)
                        best_optimizer = deepcopy(optimizer)
                        best_scheduler = deepcopy(scheduler)
                        best_score = score
                        best_results = results
                        best_epoch = epoch_idx
                        best_iter = global_iter_idx[0]
                        best_embed = embed_all
                        early_stopping_counter = 0

                        if epoch_idx == args.max_epoch - 1:
                            early_stopping_counter = args.early_stopping_threshold

                    else:
                        early_stopping_counter += 1

                    if early_stopping_counter >= args.early_stopping_threshold:
                        logger.info(f'Early stopping triggered, best epoch: {best_epoch}')
                        for k, v in best_results.items():
                            logger.info(f'Best {k}: {v:.4f}')

                        if args.save_params:
                            save_params(model=best_model,
                                        args=args,
                                        epoch_idx=best_epoch,
                                        iter_idx=best_iter,
                                        optimizer=best_optimizer,
                                        scheduler=best_scheduler)

                        if args.model['save_embed']:
                            save_embed(embed=best_embed, args=args)

                        logger.info('Process completed')
                        break

        if args.task == 'pretrain':
            if args.save_params:
                save_params(model=deepcopy(model),
                            args=args,
                            epoch_idx=epoch_idx,
                            iter_idx=global_iter_idx[0],
                            optimizer=deepcopy(optimizer),
                            scheduler=deepcopy(scheduler))

            if args.model['save_embed']:
                save_embed(embed=embed_all, args=args, type='train')
                # save_embed(embed=val_embed_all, args=args, type='val')

    elif args.model['mode'] == 'inference':
        results = single_validate(
            model,
            args.model['model_type'],
            args.task,
            val_loader,
            start_epoch,
            global_iter_idx,
            criterions=[CRITERIONS[criterion](**args.criterion[criterion]) for criterion in args.criterion],
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
    embed_all = {'patient_embed': [], 'visit_embed': [], 'code_embed': []}
    attn_mask_all = []

    for idx, data in enumerate(dataloader):

        data = data.to(device)
        optimizer.zero_grad()

        additional_data = {
            'cat_node_ids': data.cat_node_ids,
            'cat_edge_ids': data.cat_edge_ids,
            'cat_edge_index': data.cat_edge_index,
            'cat_edge_attr': data.cat_edge_attr,
            'visit_nodes': data.visit_nodes,
            'visit_node_type': data.visit_node_type,
            'ehr_nodes': data.ehr_nodes,
            'batch': data.batch,
            'batch_patient': data.batch_patient
        }

        output = model(node_ids=data.visit_node_ids,
                       edge_idx=data.global_edge_index,
                       edge_attr=data.global_edge_attr,
                       visit_times=data.visit_rel_times,
                       visit_order=data.visit_order,
                       attn_mask=data.attn_mask,
                       **additional_data)

        out = output['logits']
        prototypes = output['prototypes']
        embeddings = output['embeddings']

        attn_mask_all.append(data.attn_mask.cpu().detach())

        loss = 0.
        if task == 'pretrain':
            labels_1 = data.set_trans_label
            labels_2 = data.gru_label

            visit_mask = data.attn_mask.all(dim=-1)
            visit_mask = ~visit_mask
            visit_mask = visit_mask.unsqueeze(-1).expand(-1, -1, labels_1.size(-1))

            labels_1 = labels_1[visit_mask].view(-1, labels_1.size(-1))

            for criterion in criterions:
                    if criterion.NAME == 'binary_entropy':
                        loss += criterion(out[0], labels_1)
                        loss += criterion(out[1], labels_2)

                    elif criterion.NAME == 'contrastive_loss':
                        loss += criterion(embeddings['visit_embed'], data.attn_mask)

            metric_out = out[1]
            metric_labels = labels_2

        else:
            labels = data.labels
        
            # Main Model Finetune
            if model_type == 'base':
                for criterion in criterions:
                    if criterion.TYPE == 'prototype':
                        visit_prototypes = prototypes['visit_prototypes']
                        patient_prototypes = prototypes['patient_prototypes']
                        loss += criterion([visit_prototypes, patient_prototypes])

                    else:
                        loss += criterion(out[0], labels)
            # Baselines
            else:
                for criterion in criterions:
                    loss += criterion(out[0], labels)

            metric_out = out[0]
            metric_labels = labels

        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if task == 'los_prediction':
            probability = F.softmax(metric_out, dim=-1)
        else:
            probability = torch.sigmoid(metric_out)

        prob_all.append(probability.cpu().detach())
        target_all.append(metric_labels.cpu().detach())

        if embeddings is not None:
            for k, v in embeddings.items():
                if v is not None:
                    embed_all[k].append(v.cpu().detach())

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

    label_all = target_all

    prob_all = np.concatenate(prob_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)

    for metric in metrics:
        score = metric.calculate(prob_all, target_all)
        metric.log(score, logger, writer=writer, global_iter=global_iter_idx[0], name_prefix='train/')

    return embed_all, label_all, attn_mask_all


def single_validate(model,
                    model_type,
                    task,
                    dataloader,
                    epoch_idx,
                    global_iter_idx,
                    criterions=[],
                    metrics=[],
                    writer=None):

    model.eval()
    epoch_loss = []
    prob_all = []
    target_all = []
    embed_all = {'patient_embed': [], 'visit_embed': [], 'code_embed': []}
    attn_mask_all = []

    for _, data in enumerate(tqdm(dataloader)):
        data = data.to(device)

        with torch.no_grad():
            additional_data = {
                'cat_node_ids': data.cat_node_ids,
                'cat_edge_ids': data.cat_edge_ids,
                'cat_edge_index': data.cat_edge_index,
                'cat_edge_attr': data.cat_edge_attr,
                'visit_nodes': data.visit_nodes,
                'visit_node_type': data.visit_node_type,
                'ehr_nodes': data.ehr_nodes,
                'batch': data.batch,
                'batch_patient': data.batch_patient
            }

            output = model(node_ids=data.visit_node_ids,
                           edge_idx=data.global_edge_index,
                           edge_attr=data.global_edge_attr,
                           visit_times=data.visit_rel_times,
                           visit_order=data.visit_order,
                           attn_mask=data.attn_mask,
                           **additional_data)

            out = output['logits']
            prototypes = output['prototypes']
            embeddings = output['embeddings']

            loss = 0.
            if task == 'pretrain':
                labels_1 = data.set_trans_label
                labels_2 = data.gru_label

                visit_mask = data.attn_mask.all(dim=-1)
                visit_mask = ~visit_mask
                visit_mask = visit_mask.unsqueeze(-1).expand(-1, -1, labels_1.size(-1))

                labels_1 = labels_1[visit_mask].view(-1, labels_1.size(-1))

                for criterion in criterions:
                        if criterion.NAME == 'binary_entropy':
                            loss += criterion(out[0], labels_1)
                            loss += criterion(out[1], labels_2)

                        elif criterion.NAME == 'contrastive_loss':
                            loss += criterion(embeddings['visit_embed'], data.attn_mask)

                metric_out = out[1]
                metric_labels = labels_2

            else:
                labels = data.labels
            
                # Main Model Finetune
                if model_type == 'base':
                    for criterion in criterions:
                        if criterion.TYPE == 'prototype':
                            visit_prototypes = prototypes['visit_prototypes']
                            patient_prototypes = prototypes['patient_prototypes']
                            loss += criterion([visit_prototypes, patient_prototypes])

                        else:
                            loss += criterion(out[0], labels)
                # Baselines
                else:
                    for criterion in criterions:
                        loss += criterion(out[0], labels)

                metric_out = out[0]
                metric_labels = labels

            epoch_loss.append(loss.item())

            if task == 'los_prediction':
                probability = F.softmax(metric_out, dim=-1)
            else:
                probability = torch.sigmoid(metric_out)

            prob_all.append(probability.cpu())
            target_all.append(metric_labels.cpu())

            if embeddings is not None:
                for k, v in embeddings.items():
                    if v is not None:
                        embed_all[k].append(v.cpu())

    epoch_loss_avg = np.mean(epoch_loss)
    logger.info(f"Epoch: {epoch_idx:4d},  [{global_iter_idx[0]:5d}], Epoch Loss: {epoch_loss_avg}")
    if writer is not None:
        writer.add_scalar('val/epoch_loss', epoch_loss_avg, epoch_idx)

    label_all = target_all

    prob_all = np.concatenate(prob_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)

    results = {}
    for metric in metrics:
        score = metric.calculate(prob_all, target_all)
        metric.log(score, logger, writer=writer, global_iter=global_iter_idx[0], name_prefix='val/')
        results[metric.NAME] = score

    return results, embed_all, label_all, attn_mask_all


if __name__ == '__main__':
    args = get_args()
    main(args=args)
