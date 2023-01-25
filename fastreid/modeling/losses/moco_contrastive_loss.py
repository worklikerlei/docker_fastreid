# encoding: utf-8
"""
@author:  xiao zhou
@contact: zhouxiao17@mails.tsinghua.edu.cn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["moco_contrastive_loss", "memory_moco_contrastive_loss"]

def moco_contrastive_loss(feat, targets, T):
       
    bs_size = feat.size(0)
    feat = nn.functional.normalize(feat, dim=1)

    N_id = len(torch.unique(targets))
    N_ins = bs_size // N_id
    
    ## positive 
    dist_qk = torch.matmul(feat, feat.T)
    
    is_pos = targets.view(bs_size, 1).expand(bs_size, bs_size).eq(targets.expand(bs_size, bs_size)).float()

    s_p = dist_qk * is_pos  # (bs_size * bs_size)
    
    hard_pos = torch.eye(N_id,device=targets.device)
    for i in range(N_id):
        for j in range(N_id):
            sim_block = s_p[(N_ins*i):(N_ins*(i+1)),(N_ins*j):(N_ins*(j+1))]
            if i == j:
                v,_ = torch.min(sim_block,1)
                hard_pos[i,j] = torch.mean(v)
    
    l_pos = torch.diag(hard_pos).unsqueeze(1)# (N_id *1)
    
    ## negative        
    is_neg = targets.view(bs_size, 1).expand(bs_size, bs_size).ne(targets.expand(bs_size, bs_size)).float()

    s_n = dist_qk * is_neg + (-9999.) * (1 - is_neg) # (bs_zise * K)
    
    hard_neg = 1 - torch.zeros(N_id, bs_size, device=targets.device)
    for i in range(N_id):
        sim_block = s_n[(N_ins*i):(N_ins*(i+1)),:]
        max_s_n,_ = torch.max(sim_block,0)
        hard_neg[i,:] = max_s_n
    
    # sum_left = torch.from_numpy(np.kron(np.eye(N_id), np.ones([1,N_ins]))).float().to(is_pos.device)
    l_neg = hard_neg  # (N_id * K)
    
    nan_flag = torch.any(torch.isnan(l_neg))
    if nan_flag:
        print('NAN is here !!!!!')

    # logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)

    # apply temperature
    logits /= T

    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device = is_pos.device)
    
    criterion = nn.CrossEntropyLoss().to(is_pos.device)

    loss = criterion(logits, labels)

    return loss
    
def memory_moco_contrastive_loss(feat_q, feat_k, memory_queue, targets, memory_id, T):
       
    bs_size = feat_q.size(0)
    queue_size = memory_queue.size(0)
    feat_q = nn.functional.normalize(feat_q, dim=1)
    feat_k = nn.functional.normalize(feat_k, dim=1)
    queue = nn.functional.normalize(memory_queue, dim=1)

    N_id = len(torch.unique(targets))
    N_ins = bs_size // N_id
    
    ## positive 
    dist_qk = torch.matmul(feat_q, feat_k.T)
    
    is_pos = targets.view(bs_size, 1).expand(bs_size, bs_size).eq(targets.expand(bs_size, bs_size)).float()

    s_p = dist_qk * is_pos  # (bs_size * bs_size)
    
    hard_pos = torch.eye(N_id,device=targets.device)
    for i in range(N_id):
        for j in range(N_id):
            sim_block = s_p[(N_ins*i):(N_ins*(i+1)),(N_ins*j):(N_ins*(j+1))]
            if i == j:
                v,_ = torch.min(sim_block,1)
                hard_pos[i,j] = torch.mean(v)
    
    l_pos = torch.diag(hard_pos).unsqueeze(1)# (N_id *1)
    
    ## negative
    dist_qqu = torch.matmul(feat_q, queue.T)

    is_neg = targets.view(bs_size, 1).expand(bs_size, queue_size).ne(memory_id.expand(bs_size, queue_size)).float()

    s_n = dist_qqu * is_neg + (-9999.) * (1 - is_neg) # (bs_zise * K)
    
    N_m = memory_id.size(1)
    hard_neg = 1 - torch.zeros(N_id, N_m, device=targets.device)
    for i in range(N_id):
        sim_block = s_n[(N_ins*i):(N_ins*(i+1)),:]
        max_s_n,_ = torch.max(sim_block,0)
        hard_neg[i,:] = max_s_n
    
    l_neg = hard_neg  # (N_id * K)
    
    nan_flag = torch.any(torch.isnan(l_neg))
    if nan_flag:
        print('NAN is here !!!!!')

    # logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)

    # apply temperature
    logits /= T

    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device = is_pos.device)
    
    criterion = nn.CrossEntropyLoss().to(is_pos.device)

    loss = criterion(logits, labels)

    return loss