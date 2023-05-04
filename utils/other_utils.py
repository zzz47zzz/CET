import logging
import time
import argparse
import os
import socket
import subprocess
import json
logger = logging.getLogger("MAIN")
import numpy as np
import scipy
import torch
import torch.nn.functional as F


def get_symm_kl(noised_logits, input_logits):
        return torch.nn.KLDivLoss()(
                    F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                    F.softmax(input_logits, dim=-1, dtype=torch.float32))+\
                torch.nn.KLDivLoss()(
                   F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                   F.softmax(noised_logits, dim=-1, dtype=torch.float32))

def get_match_id(flatten_feat_train, top_k=5, max_samples=10000, metric='euclidean', largest=False):
    '''
        Compute the nearest samples id for each sample,

        Params:
            - flatten_feat_train: a matrix has dims (num_samples, hidden_dims)
            - top_k: for each sample, return the id of the top_k nearest samples
            - max_samples: number of maximum samples for computation.
                            if it is set too large, "out of memory" may happen.
            - metirc: 'euclidean' or 'cosine'
        Return:
            - match_id: a list has dims (num_samples,top_k) 
            and it represents the ids of the nearest samples of each sample.
    '''
    num_samples_all = flatten_feat_train.shape[0]
    if metric == 'euclidean':
        if num_samples_all>max_samples:
            # 2.1. calculate the L2 distance inside z0
            dist_z =  scipy.spatial.distance.cdist(flatten_feat_train,
                                    flatten_feat_train[:max_samples],
                                    'euclidean')
            dist_z = torch.tensor(dist_z)
            # 2.2. calculate distance mask: do not use itself
            mask_input = torch.clamp(torch.ones_like(dist_z)-torch.eye(num_samples_all, max_samples), 
                                    min=0)
        else:
            # 2.1. calculate the L2 distance inside z0
            dist_z = pdist(flatten_feat_train, squared=False)
            
            # 2.2. calculate distance mask: do not use itself
            mask_input = torch.clamp(torch.ones_like(dist_z)-torch.eye(num_samples_all), min=0)
    elif metric == 'cosine':
        flatten_feat_train = flatten_feat_train/torch.norm(flatten_feat_train, dim=-1).reshape(-1,1)
        if num_samples_all>max_samples:
            dist_z = torch.matmul(flatten_feat_train,flatten_feat_train[:max_samples,:].T)
            dist_z = torch.ones_like(dist_z) - dist_z
            mask_input = torch.clamp(torch.ones_like(dist_z)-torch.eye(num_samples_all, max_samples), 
                                    min=0)
        else:
            dist_z = torch.matmul(flatten_feat_train,flatten_feat_train.T)
            dist_z = torch.ones_like(dist_z) - dist_z
            mask_input = torch.clamp(torch.ones_like(dist_z)-torch.eye(num_samples_all), min=0)
    else:
        raise Exception('Invalid metric %s'%(metric))
    # 2.3 find the image meets label requirements with nearest old feature
    dist_z = mask_input.float() * dist_z
    dist_z[mask_input == 0] = float("inf")
    match_dist_matrix, match_id_matrix = torch.topk(dist_z, top_k, largest=largest, dim=1)

    # Show the average distance
    # distance_mean = torch.mean(dist_z,dim=1).reshape(-1,1)
    # topk_value = torch.topk(dist_z, k=top_k, largest=False, dim=1)[0][:,1:] # (num_samples, topk)
    # topk_ratio = topk_value/distance_mean
    # print(torch.mean(topk_ratio,dim=0))

    return dist_z, match_dist_matrix, match_id_matrix

def pdist(e, squared=False, eps=1e-12):
    '''
        Compute the L2 distance of all features

        Params:
            - e: a feature matrix has dims (num_samples, hidden_dims)
            - squared: if return the squared results of the distance
            - eps: the threshold to avoid negative distance
        Return:
            - res: a distance matrix has dims (num_samples, num_samples)
    '''
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)

def check_file(file):
    return os.path.isfile(file)

def export_config(config, path):
    param_dict = dict(vars(config))
    check_path(path)
    with open(path, 'w') as fout:
        json.dump(param_dict, fout, indent=4)

def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True

def test_data_loader_ms_per_batch(data_loader, max_steps=10000):
    start = time.time()
    n_batch = sum(1 for batch, _ in zip(data_loader, range(max_steps)))
    return (time.time() - start) * 1000 / n_batch


def params_statistic(model):
    pretrain_model_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "pretrain_model" in name: # pretrain_model params
            pretrain_model_params.append((name, param.numel(), param.device))
        else:
            other_params.append((name, param.numel(), param.device))
    num_params_pretrain_model = sum(p[1] for p in pretrain_model_params)
    num_params_other = sum(p[1] for p in other_params)
    logger.info('Total trainable param: Pretrain_model=%.3f M,  Other=%.3f M'%(
                    num_params_pretrain_model/1e6,
                    num_params_other/1e6
                ))


def print_system_info():
    logger.info('='*25+'System Info'+'='*25)
    logger.info('{0:>30}: {1}'.format('Hostname', socket.gethostname()))
    logger.info('{0:>30}: {1}'.format('Pid', os.getpid()))
    logger.info('{0:>30}: {1}'.format('Torch version', torch.__version__))
    logger.info('{0:>30}: {1}'.format('Torch cuda version', torch.version.cuda))
    logger.info('{0:>30}: {1}'.format('Cuda is available', torch.cuda.is_available()))
    logger.info('{0:>30}: {1}'.format('Cuda device count', torch.cuda.device_count()))
    logger.info('{0:>30}: {1}'.format('Cudnn version', torch.backends.cudnn.version()))

def print_basic_info(args):
    logger.info('='*25+'Experiment Info'+'='*25)
    for k, v in vars(args).items():
        logger.info('{0:>30}: {1}'.format(k, v))
