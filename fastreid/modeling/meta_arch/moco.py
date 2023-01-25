
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

from fastreid.utils.events import get_event_storage
from fastreid.utils import comm

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
                        'scale': cfg.MODEL.LOSSES.CE.SCALE,
                        'cls_type': cfg.MODEL.HEADS.CLS_LAYER
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
                        'scale': cfg.MODEL.LOSSES.MOCO.SCALE,
                        'type': cfg.MODEL.LOSSES.MOCO.TYPE
                    },
                    'mococont':{
                        'scale': cfg.MODEL.LOSSES.MOCOCONT.SCALE
                    },
                    'maxcone': {
                        'scale': cfg.MODEL.LOSSES.MAXCONE.SCALE
                    },
                    'multince': {
                        'scale': cfg.MODEL.LOSSES.MULTINCE.SCALE
                    }
                }

        self.dim = cfg.MODEL.HEADS.EMBEDDING_DIM if cfg.MODEL.HEADS.EMBEDDING_DIM \
        else cfg.MODEL.BACKBONE.FEAT_DIM
        self.num_feat_per_id = cfg.MODEL.NUM_FEAT_PER_ID

        #self.queue_ids = 8192#16384
        #self.K = self.queue_ids * self.num_feat_per_id #cfg.MODEL.HEADS.NUM_CLASSES * self.num_feat_per_id
        self.K = cfg.MODEL.HEADS.NUM_CLASSES * self.num_feat_per_id
        self.m = m
        self.T = T

        self.head_m = cfg.MODEL.HEADS.MARGIN
        self.head_s = cfg.MODEL.HEADS.SCALE

        self.mixstyle = cfg.MODEL.BACKBONE.MIXSTYLE
        self.img_per_batch = cfg.SOLVER.IMS_PER_BATCH

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
        if 'MoCoLoss' in self.loss_kwargs['loss_names'] and self.loss_kwargs.get('ce').get('cls_type') == 'Linear':
            nn.init.normal_(self.queue, std=0.01)

        # self.queue = nn.functional.normalize(self.queue, dim=0)
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
    def _dequeue_and_enqueue_N_feat_per_id(self, keys, ids, num_feat = 1):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        ids = concat_all_gather(ids)
        
        batch_size = keys.shape[0]
        unique_ids = torch.unique(ids)
        num_instance = batch_size // len(unique_ids)

        assert num_instance >= num_feat

        ptr = int(self.queue_ptr)

        unique_keys = dict()
        for i in range(batch_size):
            cur_id = ids.cpu().numpy()[i]
            if cur_id not in unique_keys:
                unique_keys[cur_id] = keys[i,:].unsqueeze(0)
            else:
                unique_keys[cur_id] = torch.cat((unique_keys[cur_id], keys[i,:].unsqueeze(0)), dim=0)
                # print(unique_keys[cur_id].shape)

        for j in range(len(unique_ids)):
            cur_id = unique_ids.cpu().numpy()[j]
            rand_rows = torch.randperm(unique_keys[cur_id].size()[0])[0:num_feat]
            if cur_id not in self.id_queue:
                if ptr + num_feat <= self.K:
                    self.id_queue[:,ptr:ptr+num_feat] = unique_ids[j]
                    self.queue[:,ptr:ptr+num_feat] = unique_keys[cur_id][rand_rows,:].T
                    ptr += num_feat
                else:
                    remain = ptr + num_feat - self.K
                    self.id_queue[:, ptr:] = unique_ids[j]
                    self.id_queue[:, :remain] = unique_ids[j]

                    self.queue[:,ptr:] = unique_keys[cur_id][rand_rows[:self.K-ptr],:].T
                    self.queue[:,:remain] = unique_keys[cur_id][rand_rows[self.K-ptr:],:].T

                    ptr = remain

            else:
                pos = np.argwhere(self.id_queue.cpu().numpy()==cur_id)
                col_num = pos[:,1]
                self.queue[:, col_num] = unique_keys[cur_id][rand_rows,:].T

        self.queue_ptr[0] = ptr
        
    @torch.no_grad()
    def _dequeue_and_enqueue_hard_feat_per_id(self, keys, ids, num_feat, weight):
        # gather keys before updating queue

        # print("hard_feat memory...")
        keys = concat_all_gather(keys)
        ids = concat_all_gather(ids)
        
        batch_size = keys.shape[0]
        unique_ids = torch.unique(ids)
        num_instance = batch_size // len(unique_ids)

        assert num_instance >= num_feat

        ptr = int(self.queue_ptr)

        logits = F.linear(keys, F.normalize(weight))
        logits = self.prepare_circle_logits(logits, ids)

        num_classes = logits.size(1)

        smooth_param = self.loss_kwargs.get('ce').get('eps')

        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            targets = torch.ones_like(log_probs)
            targets *= smooth_param / (num_classes - 1)
            targets.scatter_(1, ids.data.unsqueeze(1), (1 - smooth_param))

        loss_all = (targets * log_probs).sum(dim=1)
        for i in range(len(unique_ids)):
            index = ids == unique_ids[i]
            loss_i = loss_all[index]
            keys_i = keys[index,:]

            hard_index = torch.argsort(loss_i)[0:num_feat]
            keys_to_queue = keys_i[hard_index,:]

            cur_id = unique_ids.cpu().numpy()[i]
            if cur_id not in self.id_queue:
                if ptr + num_feat <= self.K:
                    self.id_queue[:,ptr:ptr+num_feat] = unique_ids[i]
                    self.queue[:,ptr:ptr+num_feat] = keys_to_queue.T
                    ptr += num_feat
                else:
                    remain = ptr + num_feat - self.K
                    self.id_queue[:, ptr:] = unique_ids[i]
                    self.id_queue[:, :remain] = unique_ids[i]

                    self.queue[:,ptr:] = keys_to_queue[:self.K-ptr,:].T
                    self.queue[:,:remain] = keys_to_queue[self.K-ptr:,:].T

                    ptr = remain

            else:
                pos = np.argwhere(self.id_queue.cpu().numpy()==cur_id)
                col_num = pos[:,1]
                self.queue[:, col_num] = keys_to_queue.T

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
            weight = outputs['weight']
            
            # Log prediction accuracy
            if self.mixstyle:
                targets_0 = targets[0 : self.img_per_batch//comm.get_world_size()]
                log_accuracy(pred_class_logits, targets_0)
            else:
                log_accuracy(pred_class_logits, targets)
            
        pred_features = outputs['features']  # queries: NxC
        q = pred_features #nn.functional.normalize(pred_features, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(images)

            k_input = dict()
            k_input['images'] = im_k
            k_input['targets'] = targets

            k = self.encoder_k(k_input)['features']  # keys: NxC
            # k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

                
        # dequeue and enqueue
        self._dequeue_and_enqueue_N_feat_per_id(k, targets, self.num_feat_per_id)
        # self._dequeue_and_enqueue_hard_feat_per_id(k, targets, self.num_feat_per_id, weight)


        # for i,param in enumerate(self.encoder_q.backbone.conv1.named_parameters()):
        #     if i ==0:
        #         test = param[1]
        #         print(test[0:2,0:2,0:2,0:2])

        #print(targets)
        #print(self.id_queue)

        ## compute loss
        loss_dict = {}

        # traditional reid loss
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            if self.mixstyle:
                targets_0 = targets[0 : self.img_per_batch//comm.get_world_size()]
            else:
                targets_0 = targets
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                targets_0,
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
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                q, #pred_features,
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
        
        # moco contrastive loss
        if 'MocoCont' in loss_names:
            mococont_kwargs = self.loss_kwargs.get('mococont')
            loss_dict['loss_mococont'] = moco_contrastive_loss(
                pred_features,
                targets,
                self.T
            ) * mococont_kwargs.get('scale')
        
        if 'MaxCone' in loss_names:
            maxcone_kwargs = self.loss_kwargs.get('maxcone')
            loss, orth_measure = maxcone_loss(q, targets)  # torch.tensor(-1.,device=targets.device) # 
            loss_dict['loss_maxcone'] = loss*maxcone_kwargs.get('scale')
        else:
            orth_measure = -1.
        
        ## moco memory loss
        if 'MoCoLoss' in loss_names:
            
            if self.mixstyle:
                targets_1 = targets[self.img_per_batch//comm.get_world_size():]
                q_1 = q[self.img_per_batch//comm.get_world_size():,:]
                k_1 = k[self.img_per_batch//comm.get_world_size():,:]
            else:
                targets_1 = targets
                q_1 = q
                k_1 = k
            
            moco_kwargs = self.loss_kwargs.get('moco')

            # loss_dict['loss_memory'] = self.memory_triplet_loss(q,k,targets,tri_kwargs,moco_kwargs)
            if moco_kwargs.get('type') == 'triplet':
                tri_kwargs = self.loss_kwargs.get('tri')
                loss_dict['loss_memory'] = triplet_loss(
                    #pred_features, # for original triplet
                    q_1,  # for moco memory
                    targets_1,
                    tri_kwargs.get('margin'),
                    tri_kwargs.get('norm_feat'),
                    tri_kwargs.get('hard_mining'),
                    k_1,
                    targets_1,
                    self.queue.t(),
                    self.id_queue

                ) * moco_kwargs.get('scale')
            elif moco_kwargs.get('type') == 'circle':
                circle_kwargs = self.loss_kwargs.get('circle')
                loss_dict['loss_memory'] = memory_pairwise_circleloss(
                    q_1,
                    k_1,
                    self.queue.t(),
                    targets_1,
                    self.id_queue,
                    circle_kwargs.get('margin'),
                    circle_kwargs.get('gamma')
                ) * moco_kwargs.get('scale')
            elif moco_kwargs.get('type') == 'mococont':
                loss_dict['loss_memory'] = memory_moco_contrastive_loss(
                    q_1,
                    k_1,
                    self.queue.t(),
                    targets_1,
                    self.id_queue,
                    self.T
                ) * moco_kwargs.get('scale')
            elif moco_kwargs.get('type') == 'maxcone':
                loss_dict['loss_memory'] = memory_maxcone_loss(
                    q_1,
                    k_1,
                    self.queue.t(),
                    targets_1,
                    self.id_queue,
                    self.num_feat_per_id
                ) * moco_kwargs.get('scale')

        storage = get_event_storage()

        return loss_dict, orth_measure

    def memory_triplet_loss(self, q,k,targets,tri_kwargs,moco_kwargs):

        l1 = triplet_loss(q,targets.unsqueeze(0),tri_kwargs.get('margin'),tri_kwargs.get('norm_feat'),tri_kwargs.get('hard_mining'),k,targets.unsqueeze(0), self.queue.t(),self.id_queue)
        l2 = triplet_loss(k,targets.unsqueeze(0),tri_kwargs.get('margin'),tri_kwargs.get('norm_feat'),tri_kwargs.get('hard_mining'),self.queue.t(), self.id_queue, q, targets.unsqueeze(0))
        l3 = triplet_loss(self.queue.t(),self.id_queue,tri_kwargs.get('margin'),tri_kwargs.get('norm_feat'),tri_kwargs.get('hard_mining'),q,targets.unsqueeze(0), k,targets.unsqueeze(0))

        return (l1+l2+l3)*moco_kwargs.get('scale')


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

    def prepare_circle_logits(self, logits, targets):
        alpha_p = torch.clamp_min(-logits.detach() + 1 + self.head_m, min=0.)
        alpha_n = torch.clamp_min(logits.detach() + self.head_m, min=0.)
        delta_p = 1 - self.head_m
        delta_n = self.head_m

        logits_p = alpha_p * (logits - delta_p)
        logits_n = alpha_n * (logits - delta_n)

        # When use model parallel, there are some targets not in class centers of local rank
        index = torch.where(targets != -1)[0]
        if len(index) > 0:
            m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device, dtype=logits.dtype)
            m_hot.scatter_(1, targets[index, None], 1)
            logits[index] = logits_p[index] * m_hot + logits_n[index] * (1 - m_hot)

        neg_index = torch.where(targets == -1)[0]
        logits[neg_index] = logits_n[neg_index]

        logits.mul_(self.head_s)

        return logits


    def memory_circle_loss(self, cls_outputs, feat_q, targets, weight, ce_kwargs, use_memory = False, extend_sample = 0, show_sn = False):
        
        mean_s_n = -2
        mean_sigma = -1

        if ce_kwargs.get('cls_type') != 'Linear':
            feat_queue = nn.functional.normalize(self.queue, dim=0)
            feat_q = nn.functional.normalize(feat_q, dim=1)
        else:
            feat_queue = self.queue

        if not extend_sample:

            sim_neg = torch.matmul(feat_q, feat_queue)
            is_neg = targets.view(self.bs_size, 1).expand(self.bs_size, self.K).ne(self.id_queue.expand(self.bs_size, self.K))

            if torch.min(is_neg.view(-1)) == 1:
                s_n = torch.zeros(self.bs_size, self.K, device = targets.device)
            else:
                s_n = torch.zeros(self.bs_size, self.K - self.num_feat_per_id, device = targets.device)
            
            for i in range(is_neg.shape[0]):
                s_n[i,:] = sim_neg[i, is_neg[i,:] == True]
            
            median_s_n, _ = torch.median(s_n,dim=1)
            mean_s_n = torch.mean(median_s_n)

            sigma = torch.std(s_n, dim = 1)
            mean_sigma = torch.mean(sigma)

            if ce_kwargs.get('cls_type') == 'Linear':
                s_n.mul_(self.head_s)
            else:
                s_n = self.prepare_circle_logits(s_n, -1*torch.ones(s_n.size()[0], device= targets.device))

            cls_outputs_memory = torch.cat((cls_outputs, s_n), dim=1)
        
        else:
            # print('extend_sample!')
            logits = F.linear(feat_queue.T, F.normalize(weight))
            index = torch.where(self.id_queue.squeeze().type_as(targets) > 0)[0]
            memory_targets = self.id_queue.squeeze().type_as(targets)[index]
            logits = self.prepare_circle_logits(logits[index], memory_targets)

            cls_outputs_memory = torch.cat((cls_outputs, logits), dim=0)
            targets = torch.cat((targets, memory_targets))

        if not use_memory:
            cls_outputs_memory = cls_outputs

        loss_circle = cross_entropy_loss(cls_outputs_memory,targets,ce_kwargs.get('eps'),ce_kwargs.get('alpha')) * ce_kwargs.get('scale')

        if show_sn:
            return loss_circle, mean_s_n, mean_sigma
        else:
            return loss_circle

    def max_nuc_norm(self, feat_q, feat_k, targets):

        feat_q = nn.functional.normalize(feat_q, dim=1)
        # feat_k = nn.functional.normalize(feat_k, dim=1)

        N_id = len(torch.unique(targets))
        N_ins = self.bs_size // N_id
        
        sim_qq = torch.matmul(feat_q, feat_q.T)

        sum_left = torch.from_numpy(np.kron(np.eye(N_id), np.ones([1,N_ins]))).float().to(targets.device)
        sum_right = torch.from_numpy(np.kron(np.eye(N_id),np.ones([N_ins,1]))).float().to(targets.device)

        scale = 8.0

        exp_sim = torch.log(1 + torch.exp(sim_qq*scale))
        # exp_sim = torch.exp(sim_q_m*16)

        exp_sim = exp_sim * (1 - torch.eye(feat_q.size()[0], device=targets.device)) # + alpha * torch.eye(feat_q.size()[0], device=targets.device)

        mean_exp_sim = torch.matmul(sum_left, exp_sim) 
        
        N_class = self.K//self.num_feat_per_id
        # sum_right = torch.from_numpy(np.kron(np.eye(N_class),np.ones([self.num_feat_per_id,1])/self.num_feat_per_id)).float().to(targets.device)
        
        mean_exp_sim = torch.matmul(mean_exp_sim, sum_right) #(N_id * N_id)

        pos_sim = torch.diag(torch.diag(mean_exp_sim))/(N_id-1)/N_id
        neg_sim = mean_exp_sim * (1 - torch.eye(N_id, device=targets.device))/N_id/N_id
        
        v,index = torch.max(neg_sim, 1)
        alpha_mat = torch.diag(v.detach())
        alpha_pos_sim = pos_sim + alpha_mat

        mat_sim = alpha_pos_sim + neg_sim

        mat_sim = nn.functional.normalize(mat_sim, dim = 1)

        # if torch.isnan(mean_exp_sim).any():
        #     test = 1

        # print(torch.sum(mean_exp_sim,dim = 1))
        try:
            singlar_value = torch.linalg.svdvals(mat_sim.float())
        except:                     # torch.svd may have convergence issues for GPU and CPU.
            singlar_value = torch.linalg.svdvals(mat_sim.float() + 1e-4*mat_sim.float().mean()*torch.rand(mat_sim.shape[0],mat_sim.shape[1],device=targets.device))

        # U,singlar_value,V = torch.linalg.svd(mean_exp_sim.float())
        # if U[9,9] == 0:
        #     test = 1
        maxcone = torch.prod(singlar_value)

        orth_mat = pos_sim.detach() + neg_sim.detach()
        nuc_norm = torch.linalg.norm(orth_mat.float(),'nuc')
        f_norm = torch.linalg.norm(orth_mat.detach(),'fro')
        rank =  torch.tensor(N_id,dtype=torch.float32,device=targets.device)

        orth_measure = (nuc_norm - f_norm)/(rank - torch.sqrt(rank))/f_norm
    
        return maxcone, orth_measure

    def hard_max_cone(self, feat_q, feat_k, targets):

        feat_q = nn.functional.normalize(feat_q, dim=1)
        # feat_k = nn.functional.normalize(feat_k, dim=1)

        N_id = len(torch.unique(targets))
        N_ins = self.bs_size // N_id
        
        sim_qq = torch.matmul(feat_q, feat_q.T)

        scale = 16.0

        exp_sim = torch.log(1 + torch.exp(sim_qq*scale))

        hard_pos = torch.eye(N_id,device=targets.device)
        hard_neg = 1 - torch.eye(N_id,device=targets.device)
        for i in range(N_id):
            for j in range(N_id):
                sim_block = exp_sim[(N_ins*i):(N_ins*(i+1)),(N_ins*j):(N_ins*(j+1))]
                if i == j:
                    v,_ = torch.min(sim_block,1)
                    hard_pos[i,j] = torch.mean(v)
                else:
                    v,_ = torch.max(sim_block,1)
                    hard_neg[i,j] = torch.mean(v)
        
        v,_ = torch.max(hard_neg, 1)
        alpha_mat = torch.diag(v.detach())
        alpha_hard_pos = hard_pos + 2*alpha_mat

        hard_sim = alpha_hard_pos + hard_neg

        hard_sim = nn.functional.normalize(hard_sim, dim = 1)
        #hard_sim = nn.functional.normalize(hard_sim, p=1,dim = 1)

        if torch.isnan(hard_sim).any():
            print('nan is here!!!!!!!!')

        # print(torch.sum(mean_exp_sim,dim = 1))
        try:
            #singlar_value = torch.linalg.svdvals(hard_sim.float())
            _,singular_value,_ = torch.svd(hard_sim.float())
        except:                     # torch.svd may have convergence issues for GPU and CPU.
            #singular_value = torch.linalg.svdvals(hard_sim.float() + 1e-4*hard_sim.float().mean()*torch.rand(hard_sim.shape[0],hard_sim.shape[1],device=targets.device))
            _,singular_value,_ = torch.svd(hard_sim.float() + 1e-4*hard_sim.float().mean()*torch.rand(hard_sim.shape[0],hard_sim.shape[1],device=targets.device))

        #print(singular_value)    
        # U,singlar_value,V = torch.linalg.svd(mean_exp_sim.float())
        # if U[9,9] == 0:
        #     test = 1
        maxcone = torch.prod(singular_value)

        orth_mat = hard_pos.detach() + hard_neg.detach()
        #nuc_norm = torch.linalg.norm(orth_mat.float(),'nuc')
        nuc_norm = torch.norm(orth_mat.float(),'nuc')
        #f_norm = torch.linalg.norm(orth_mat.detach(),'fro')
        f_norm = torch.norm(orth_mat.detach(),'fro')
        rank =  torch.tensor(N_id,dtype=torch.float32,device=targets.device)

        orth_measure = (nuc_norm - f_norm)/(rank - torch.sqrt(rank))/f_norm
    
        return maxcone, orth_measure

    def soft_max_cone(self, feat_q, feat_k, targets):
        
        feat_q = nn.functional.normalize(feat_q, dim=1)
        # feat_k = nn.functional.normalize(feat_k, dim=1)

        N_id = len(torch.unique(targets))
        N_ins = self.bs_size // N_id
        
        scale = 8.0
        # sim_qq = scale*F.softplus(torch.matmul(feat_q, feat_q.T), beta = scale) #scale 16, 2*max_detach, map 66.69
        sim_qq = torch.matmul(feat_q, feat_q.T)
        sf_sim_qq = torch.exp(scale*sim_qq)

        hard_pos = torch.eye(N_id,device=targets.device)
        hard_neg = 1 - torch.eye(N_id,device=targets.device)
        for i in range(N_id):
            for j in range(N_id):
                logits = sf_sim_qq[(N_ins*i):(N_ins*(i+1)),(N_ins*j):(N_ins*(j+1))]

                if i == j:
                    soft_hard = F.softplus(-1*torch.logsumexp(-1.*logits,dim=1))
                    v,_ = torch.min(logits,1)
                    hard_pos[i,j] = soft_hard.mean()

                else:
                    soft_hard = F.softplus(torch.logsumexp(logits,dim = 1))
                    v,_ = torch.max(logits,1)
                    hard_neg[i,j] = soft_hard.mean()
        
        max_neg,_ = torch.max(hard_neg, dim = 1)
        max_neg_mat = torch.diag(max_neg)
        alpha_hard_pos = hard_pos + 2*max_neg_mat.detach()

        hard_sim = alpha_hard_pos + hard_neg

        ## normalize by L2-norm
        norm_hard_sim = nn.functional.normalize(hard_sim, dim = 1)
        # norm_hard_sim = nn.functional.normalize(hard_sim, p=1,dim = 1)

        try:
            _,singular_value,_ = torch.svd(norm_hard_sim.float())
        except:                     # torch.svd may have convergence issues for GPU and CPU.
            _,singular_value,_ = torch.svd(norm_hard_sim.float() + 1e-4*norm_hard_sim.float().mean()*torch.rand(norm_hard_sim.shape[0],norm_hard_sim.shape[1],device=targets.device))

        maxcone = torch.prod(singular_value)

        orth_mat = hard_pos.detach() + hard_neg.detach()
        nuc_norm = torch.norm(orth_mat.float(),'nuc')
        f_norm = torch.norm(orth_mat.detach(),'fro')
        rank =  torch.tensor(N_id,dtype=torch.float32,device=targets.device)

        orth_measure = (nuc_norm - f_norm)/(rank - torch.sqrt(rank))/f_norm

        return maxcone, orth_measure

    def multince_loss(self, feat_q, feat_k, targets):
       
        norm_feat_q = nn.functional.normalize(feat_q, dim=1)
        norm_feat_k = nn.functional.normalize(feat_k, dim=1)
        norm_queue = nn.functional.normalize(self.queue, dim=1)

        N_id = len(torch.unique(targets))
        N_ins = self.bs_size // N_id

        sum_left = torch.from_numpy(np.kron(np.eye(N_id), np.ones([1,N_ins]))).float().to(targets.device)

        center_q = nn.functional.normalize(torch.matmul(sum_left/N_ins, feat_q),dim = 1)

        sim_qc = torch.matmul(norm_feat_q, center_q.T)
        pos_sim_qc = torch.exp(-1*sim_qc*10)

        pos_exp_sim = torch.matmul(sum_left/N_ins, pos_sim_qc)

        pos_mask = torch.eye(N_id,device=targets.device)
        logits_p = torch.diag(pos_exp_sim*pos_mask)

        neg_sim_qc = torch.exp(sim_qc*10)
        neg_exp_sim = torch.matmul(sum_left/N_ins, neg_sim_qc)
        logits_n = torch.sum(neg_exp_sim, dim=1)

        pos_loss = torch.log(1+logits_p)
        neg_loss = torch.log(1+logits_n)

        loss = torch.mean(pos_loss)+ torch.mean(neg_loss)

        return loss


    def moco_contrastive_loss(self, feat_q, feat_k, memory_queue, targets, memory_id):
        feat_q = nn.functional.normalize(feat_q, dim=1)
        feat_k = nn.functional.normalize(feat_k, dim=1)
        queue = nn.functional.normalize(memory_queue, dim=1)

        N_id = len(torch.unique(targets))
        N_ins = self.bs_size // N_id

        ## positive 
        dist_qk = torch.matmul(feat_q, feat_k.T)

        is_pos = targets.view(self.bs_size, 1).expand(self.bs_size, self.bs_size).eq(targets.expand(self.bs_size, self.bs_size)).float()
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

        is_neg = targets.view(self.bs_size, 1).expand(self.bs_size, self.K).ne(memory_id.expand(self.bs_size, self.K)).float()

        s_n = dist_qqu * is_neg + (-9999.) * (1 - is_neg) # (bs_zise * K)

        N_m = memory_id.size(1)
        hard_neg = torch.zeros(N_id, N_m, device=targets.device)
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
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device = is_pos.device)

        criterion = nn.CrossEntropyLoss().to(is_pos.device)

        loss = criterion(logits, labels)

        return loss

    def hard_moco_loss(self, feat_q, feat_k, targets):
       
        #norm_feat_q = nn.functional.normalize(feat_q, dim=1)
        #norm_feat_k = nn.functional.normalize(feat_k, dim=1)
        #norm_queue = nn.functional.normalize(self.queue, dim=1)

        ## positive 
        dist_qk = torch.matmul(feat_q, feat_k.T)
        
        is_pos = targets.view(self.bs_size, 1).expand(self.bs_size, self.bs_size).eq(targets.expand(self.bs_size, self.bs_size)).float()
        # Mask scores related to themselves
        same_indx = torch.eye(self.bs_size, self.bs_size, device=is_pos.device)
        is_pos = is_pos - same_indx
        s_p = dist_qk * is_pos + (1-is_pos)*10 # (bs_size * bs_size)
        
        N_id = len(torch.unique(targets))
        N_ins = self.bs_size // N_id
        sum_left = torch.from_numpy(np.kron(np.eye(N_id), np.ones([1,N_ins]))).float().to(is_pos.device)

        sum_temp = torch.matmul(sum_left, torch.exp(-10*s_p))
        
        sum_right = torch.from_numpy(np.kron(np.eye(N_id),np.ones([N_ins,1]))).float().to(is_pos.device)
        sum_s_p = torch.matmul(sum_temp, sum_right)  #(N_id * N_id)
        
        l_pos = -1.0/10.*torch.log(torch.diag(sum_s_p)).unsqueeze(1)# (N_id *1)
        

        ## negative
        dist_qqu = torch.matmul(feat_q, self.queue)
        
        is_pos = targets.view(self.bs_size, 1).expand(self.bs_size, self.K).eq(self.id_queue.expand(self.bs_size, self.K)).float()
        is_neg = targets.view(self.bs_size, 1).expand(self.bs_size, self.K).ne(self.id_queue.expand(self.bs_size, self.K)).float()
        
        #print(targets)
        #print(self.id_queue)
        #print(self.id_queue*is_neg[0,:])
        #time.sleep(10)

        s_n = dist_qqu * is_neg + (-10.) * (1 - is_neg) # (bs_zise * K)
        
        sum_left = torch.from_numpy(np.kron(np.eye(N_id), np.ones([1,N_ins]))).float().to(is_pos.device)
        #l_neg = torch.matmul(sum_left, s_n) / N_ins  # (N_id * K)
        l_neg = 1./10.*torch.matmul(sum_left, torch.exp(10*s_n))  # (N_id * K)
        
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


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class MocoSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(MocoSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features_q, features_k, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        assert features_q.shape == features_k.shape
        device = (torch.device('cuda')
                  if features_q.is_cuda
                  else torch.device('cpu'))

        if len(features_q.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features_q.shape) > 3:
            features_q = features_q.view(features_q.shape[0], features_q.shape[1], -1)
            features_k = features_k.view(features_k.shape[0], features_k.shape[1], -1)

        batch_size = features_q.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features_k.shape[1]
        contrast_feature = torch.cat(torch.unbind(features_k, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features_q[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = torch.cat(torch.unbind(features_q, dim=1), dim=0)
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
