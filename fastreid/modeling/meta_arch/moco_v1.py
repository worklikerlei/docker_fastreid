
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# from fastreid.modeling.meta_arch.baseline import Baseline
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
# from torchvision.models import resnet50

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses.utils import concat_all_gather
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY

from fastreid.layers import get_norm
import time

@META_ARCH_REGISTRY.register()
class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, cfg, m=0.999, T=0.07):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        self.has_extra_bn = cfg.MODEL.BACKBONE.EXTRA_BN
        if self.has_extra_bn:
            self.heads_extra_bn = get_norm(cfg.MODEL.BACKBONE.NORM, cfg.MODEL.BACKBONE.FEAT_DIM)
        
        self.loss_kwargs = {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    },
                    'moco': {
                        'scale': cfg.MODEL.LOSSES.MOCO.SCALE
                    }
                }

        self.dim = cfg.MODEL.HEADS.EMBEDDING_DIM if cfg.MODEL.HEADS.EMBEDDING_DIM \
        else cfg.MODEL.BACKBONE.FEAT_DIM
        self.K = cfg.MODEL.QUEUE_SIZE
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension

        # self.backbone = build_backbone(cfg)
        # self.head = build_heads(cfg)
        import collections
        self.encoder_q = nn.Sequential(collections.OrderedDict([('backbone',build_backbone(cfg)),('head',build_heads(cfg))]))
        self.encoder_k = nn.Sequential(collections.OrderedDict([('backbone',build_backbone(cfg)),('head',build_heads(cfg))]))
        
        # if mlp:  # hack: brute-force replacement
        #     dim_mlp = self.encoder_q.fc.weight.shape[1]
        #     self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), 
        #                                       nn.ReLU(), self.encoder_q.fc)
        #     self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), 
        #                                       nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("id_queue", -1*torch.ones(1, self.K))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    # def __init__(self, cfg, m=0.999, T=0.07):
    #     """
    #     dim: feature dimension (default: 128)
    #     K: queue size; number of negative keys (default: 65536)
    #     m: moco momentum of updating key encoder (default: 0.999)
    #     T: softmax temperature (default: 0.07)
    #     """
    #     super().__init__(cfg)
    @property
    def device(self):
        return self.pixel_mean.device
        

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_v0(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys,ids):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        ids = concat_all_gather(ids)
       
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            self.id_queue[:, ptr:ptr + batch_size] = ids
            ptr = ptr + batch_size
        else:
            # p_str = f"prt: {ptr}; bs: {batch_size} "
            remain = ptr + batch_size - self.K
            self.queue[:, ptr:] = keys[:self.K - ptr].T
            self.queue[:, :remain] = keys[self.K - ptr:].T
            
            self.id_queue[:, ptr:] = ids[:self.K - ptr]
            self.id_queue[:, :remain] = ids[self.K - ptr:]
            
            ptr = remain
            # p_str += f"remain: {remain}, new ptr: {ptr}"
            # print(p_str)

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]

        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, data):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        targets = data['targets']
        self.bs_size,_,_,_ = data['images'].size()
        images = self.preprocess_image(data)

        # backbone_outputs = self.backbone(images)
        # outputs = self.head(backbone_outputs, targets)


        moco_input = dict()
        moco_input['images'] = images
        moco_input['targets'] = targets
        
        if not self.training:
            return self.encoder_q(moco_input)

        outputs = self.encoder_q(moco_input)

        # compute query features
        if len(outputs) > 1:
            # fmt: off
            pred_class_logits = outputs['pred_class_logits'].detach()
            cls_outputs       = outputs['cls_outputs']
            
            # Log prediction accuracy
            log_accuracy(pred_class_logits, targets)
            
        pred_features = outputs['features']  # queries: NxC
        q = nn.functional.normalize(pred_features, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(images)

            k_input = dict()
            k_input['images'] = im_k
            k_input['targets'] = targets

            k = self.encoder_k(k_input)['features']  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, targets)
        flag = -1 in self.id_queue


        #print(targets)
        #print(self.id_queue)

        ## compute loss
        loss_dict = {}

        # traditional reid loss
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                targets,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(
                #pred_features, # for original triplet
                q,  # for moco memory
                targets,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining'),
                k,
                self.queue,
                self.id_queue
            ) * tri_kwargs.get('scale')

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                targets,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                targets,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')
        ## moco loss
        if 'MoCoLoss' in loss_names:
            moco_kwargs = self.loss_kwargs.get('moco')
            loss_dict['loss_moco'] = self.moco_loss(q, k, targets)*moco_kwargs.get('scale')

        return loss_dict

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def moco_loss(self, feat_q, feat_k, targets):
       
        #norm_feat_q = nn.functional.normalize(feat_q, dim=1)
        #norm_feat_k = nn.functional.normalize(feat_k, dim=1)
        #norm_queue = nn.functional.normalize(self.queue, dim=1)

        ## positive 
        dist_qk = torch.matmul(norm_feat_q, norm_feat_k.T)
        
        is_pos = targets.view(self.bs_size, 1).expand(self.bs_size, self.bs_size).eq(targets.expand(self.bs_size, self.bs_size)).float()
        # Mask scores related to themselves
        same_indx = torch.eye(self.bs_size, self.bs_size, device=is_pos.device)
        is_pos = is_pos - same_indx
        s_p = dist_qk * is_pos  # (bs_size * bs_size)
        
        N_id = len(torch.unique(targets))
        N_ins = self.bs_size // N_id
        sum_left = torch.from_numpy(np.kron(np.eye(N_id), np.ones([1,N_ins]))).float().to(is_pos.device)

        sum_temp = torch.matmul(sum_left, s_p)
        
        sum_right = torch.from_numpy(np.kron(np.eye(N_id),np.ones([N_ins,1]))).float().to(is_pos.device)
        sum_s_p = torch.matmul(sum_temp, sum_right)  #(N_id * N_id)
        
        l_pos = torch.diag(sum_s_p).unsqueeze(1) / (N_ins*N_ins-N_ins) # (N_id *1)
        

        ## negative
        dist_qqu = torch.matmul(norm_feat_q, norm_queue)
        
        is_pos = targets.view(self.bs_size, 1).expand(self.bs_size, self.K).eq(self.id_queue.expand(self.bs_size, self.K)).float()
        is_neg = targets.view(self.bs_size, 1).expand(self.bs_size, self.K).ne(self.id_queue.expand(self.bs_size, self.K)).float()
        
        #print(targets)
        #print(self.id_queue)
        #print(self.id_queue*is_neg[0,:])
        #time.sleep(10)

        s_n = dist_qqu * is_neg + (-9999.) * (1 - is_neg) # (bs_zise * K)
        
        sum_left = torch.from_numpy(np.kron(np.eye(N_id), np.ones([1,N_ins]))).float().to(is_pos.device)
        l_neg = torch.matmul(sum_left, s_n) / N_ins  # (N_id * K)
        
        nan_flag = torch.any(torch.isnan(l_neg))
        if nan_flag:
            print('NAN is here !!!!!')


        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device = is_pos.device)
        
        criterion = nn.CrossEntropyLoss().to(is_pos.device)
  
        loss = criterion(logits, labels)

        return loss


# # encoding: utf-8
# """
# @author:  xingyu liao
# @contact: sherlockliao01@gmail.com
# """

# import torch
# import torch.nn.functional as F
# from torch import nn

# from fastreid.modeling.losses.utils import concat_all_gather
# from fastreid.utils import comm
# from .baseline import Baseline
# from .build import META_ARCH_REGISTRY


# @META_ARCH_REGISTRY.register()
# class MoCo(Baseline):
#     def __init__(self, cfg):
#         super().__init__(cfg)

#         dim = cfg.MODEL.HEADS.EMBEDDING_DIM if cfg.MODEL.HEADS.EMBEDDING_DIM \
#             else cfg.MODEL.BACKBONE.FEAT_DIM
#         size = cfg.MODEL.QUEUE_SIZE
#         self.memory = Memory(dim, size)

#     def losses(self, outputs, gt_labels):
#         """
#         Compute loss from modeling's outputs, the loss function input arguments
#         must be the same as the outputs of the model forwarding.
#         """
#         # regular reid loss
#         loss_dict = super().losses(outputs, gt_labels)

#         # memory loss
#         pred_features = outputs['features']
#         loss_mb = self.memory(pred_features, gt_labels)
#         loss_dict['loss_mb'] = loss_mb
#         return loss_dict


# class Memory(nn.Module):
#     """
#     Build a MoCo memory with a queue
#     https://arxiv.org/abs/1911.05722
#     """

#     def __init__(self, dim=512, K=65536):
#         """
#         dim: feature dimension (default: 128)
#         K: queue size; number of negative keys (default: 65536)
#         """
#         super().__init__()
#         self.K = K

#         self.margin = 0.25
#         self.gamma = 32

#         # create the queue
#         self.register_buffer("queue", torch.randn(dim, K))
#         self.queue = F.normalize(self.queue, dim=0)

#         self.register_buffer("queue_label", torch.zeros((1, K), dtype=torch.long))
#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

#     @torch.no_grad()
#     def _dequeue_and_enqueue(self, keys, targets):
#         # gather keys/targets before updating queue
#         if comm.get_world_size() > 1:
#             keys = concat_all_gather(keys)
#             targets = concat_all_gather(targets)
#         else:
#             keys = keys.detach()
#             targets = targets.detach()

#         batch_size = keys.shape[0]

#         ptr = int(self.queue_ptr)
#         assert self.K % batch_size == 0  # for simplicity

#         # replace the keys at ptr (dequeue and enqueue)
#         self.queue[:, ptr:ptr + batch_size] = keys.T
#         self.queue_label[:, ptr:ptr + batch_size] = targets
#         ptr = (ptr + batch_size) % self.K  # move pointer

#         self.queue_ptr[0] = ptr

#     def forward(self, feat_q, targets):
#         """
#         Memory bank enqueue and compute metric loss
#         Args:
#             feat_q: model features
#             targets: gt labels

#         Returns:
#         """
#         # normalize embedding features
#         feat_q = F.normalize(feat_q, p=2, dim=1)
        
#         # dequeue and enqueue
#         self._dequeue_and_enqueue(feat_q.detach(), targets)

#         # compute loss
#         loss = self._pairwise_cosface(feat_q, targets)
        
#         return loss

#     def _pairwise_cosface(self, feat_q, targets):
#         dist_mat = torch.matmul(feat_q, self.queue)

#         N, M = dist_mat.size()  # (bsz, memory)
#         is_pos = targets.view(N, 1).expand(N, M).eq(self.queue_label.expand(N, M)).float()
#         is_neg = targets.view(N, 1).expand(N, M).ne(self.queue_label.expand(N, M)).float()

#         # # Mask scores related to themselves
#         # same_indx = torch.eye(N, N, device=is_pos.device)
#         # other_indx = torch.zeros(N, M - N, device=is_pos.device)
#         # same_indx = torch.cat((same_indx, other_indx), dim=1)
#         # is_pos = is_pos - same_indx

#         s_p = dist_mat * is_pos
#         s_n = dist_mat * is_neg

#         logit_p = -self.gamma * s_p + (-99999999.) * (1 - is_pos)
#         logit_n = self.gamma * (s_n + self.margin) + (-99999999.) * (1 - is_neg)

#         loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

#         return loss
