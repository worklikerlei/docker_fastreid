# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import time
import torch
import torch.nn.functional as F

from .utils import euclidean_dist, cosine_dist, concat_all_gather
from fastreid.utils import comm


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N]
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N]
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

    return dist_ap, dist_an


def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2

    is_pos = is_pos
    is_neg = is_neg
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


def triplet_loss(embedding, targets, margin, norm_feat, hard_mining, k_embedding=None, k_id = None, queue_embedding = None, id_queue = None):
    r"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""
    #print(id_queue.view(-1))
    #print(min(id_queue.view(-1)))
    if queue_embedding != None and id_queue != None: #and min(id_queue.view(-1)) != -1:
        #print(queue_embedding.shape)
        #print(id_queue.shape)
        #print('No -1 in id_queue')
        assert queue_embedding.size(0) == id_queue.size(1)
        
        neg_embedding = queue_embedding
        neg_id = id_queue

    elif k_embedding is not None and k_id != None:

        assert k_embedding.size(0) == k_id.size(1)

        neg_embedding = k_embedding
        neg_id = k_id
    else:
        #print('-1 in id_queue')
        neg_embedding = embedding
        neg_id = targets

    if k_embedding is not None:
        pos_embedding = k_embedding
        pos_id = k_id
    else:
        pos_embedding = embedding
        pos_id = targets

    #print(embedding.shape)
    #print(queue_embedding.shape)
    

    if norm_feat:
        dist_mat_pos = cosine_dist(embedding, pos_embedding)
        dist_mat_neg = cosine_dist(embedding, neg_embedding)
        #print(torch.sum(embedding**2, dim=1))
        #print(torch.sum(pos_embedding**2, dim=1))
        #print(torch.sum(neg_embedding**2, dim=1))
    else:
        dist_mat_pos = euclidean_dist(embedding, pos_embedding)
        dist_mat_neg = euclidean_dist(embedding, neg_embedding)

    # For distributed training, gather all features from different process.
    # if comm.get_world_size() > 1:
    #     all_embedding = torch.cat(GatherLayer.apply(embedding), dim=0)
    #     all_targets = concat_all_gather(targets)
    # else:
    #     all_embedding = embedding
    #     all_targets = targets

    N_pos = dist_mat_pos.size(0)
    K_pos = dist_mat_pos.size(1)
    is_pos = targets.view(N_pos, 1).expand(N_pos, K_pos).eq(pos_id.view(1, K_pos).expand(N_pos, K_pos)).float()
    #print(is_pos)

    N_neg = dist_mat_neg.size(0)
    K_neg = dist_mat_neg.size(1)
    is_neg = targets.view(N_neg, 1).expand(N_neg, K_neg).ne(neg_id.view(1, K_neg).expand(N_neg, K_neg)).float()
    #print(is_neg)

    #print(targets)
    #print(id_queue)
    #print(is_neg)
    #time.sleep(10)

    if hard_mining:
        dist_ap, _ = hard_example_mining(dist_mat_pos, is_pos, 1-is_pos)
        _, dist_an = hard_example_mining(dist_mat_neg, 1-is_neg, is_neg)
    else:
        dist_ap, _ = weighted_example_mining(dist_mat_pos, is_pos, 1-is_pos)
        _, dist_an = weighted_example_mining(dist_mat_neg, 1-is_neg, is_neg)

    y = dist_an.new().resize_as_(dist_an).fill_(1)

    if margin > 0:
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    else:
        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        # fmt: off
        if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
        # fmt: on

    return loss
