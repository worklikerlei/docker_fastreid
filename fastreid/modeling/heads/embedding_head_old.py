# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.config import configurable
from fastreid.layers import *
from fastreid.layers import pooling, any_softmax
from fastreid.layers.weight_init import weights_init_kaiming
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class EmbeddingHead(nn.Module):
    """
    EmbeddingHead perform all feature aggregation in an embedding task, such as reid, image retrieval
    and face recognition

    It typically contains logic to

    1. feature aggregation via global average pooling and generalized mean pooling
    2. (optional) batchnorm, dimension reduction and etc.
    2. (in training only) margin-based softmax logits computation
    """

    @configurable
    def __init__(
            self,
            *,
            feat_dim,
            embedding_dim,
            num_classes,
            neck_feat,
            pool_type,
            cls_type,
            scale,
            margin,
            with_bnneck,
            norm_type,
            cls_enable,
            loss_names,
            mixstyle
    ):
        """
        NOTE: this interface is experimental.

        Args:
            feat_dim:
            embedding_dim:
            num_classes:
            neck_feat:
            pool_type:
            cls_type:
            scale:
            margin:
            with_bnneck:
            norm_type:

            cls_enable:
        """
        super().__init__()

        # Pooling layer
        assert hasattr(pooling, pool_type), "Expected pool types are {}, " \
                                            "but got {}".format(pooling.__all__, pool_type)
        self.pool_layer = getattr(pooling, pool_type)()

        self.neck_feat = neck_feat

        
        if with_bnneck:
            self.neck_bn = get_norm(norm_type, feat_dim, bias_freeze=True)
        
        neck = []
        if embedding_dim > 0:
            #neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            neck.append(nn.Linear(feat_dim, embedding_dim))
            neck.append(nn.ReLU())
            feat_dim = embedding_dim
            
        # if 'MaxCone' in loss_names:
        #     neck.append(nn.ReLU())
        
        self.bottleneck = nn.Sequential(*neck)

        # Classification head
        
        self.cls_enable = cls_enable
        
        if self.cls_enable:
            assert hasattr(any_softmax, cls_type), "Expected cls types are {}, " \
                                               "but got {}".format(any_softmax.__all__, cls_type)
            self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim))
            self.cls_layer = getattr(any_softmax, cls_type)(num_classes, scale, margin)

        self.reset_parameters()

        self.mixstyle = mixstyle


    def reset_parameters(self) -> None:
        #print('I am here!!!!')
        self.bottleneck.apply(weights_init_kaiming)
        if self.cls_enable:
            nn.init.normal_(self.weight, std=0.01)

    @classmethod
    def from_config(cls, cfg):
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        scale         = cfg.MODEL.HEADS.SCALE
        margin        = cfg.MODEL.HEADS.MARGIN
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM
        loss_names    = cfg.MODEL.LOSSES.NAME
        mixstyle      = cfg.MODEL.BACKBONE.MIXSTYLE
        
        cls_enable = cfg.MODEL.HEADS.CLS_ENABLE
        # fmt: on
        return {
            'feat_dim': feat_dim,
            'embedding_dim': embedding_dim,
            'num_classes': num_classes,
            'neck_feat': neck_feat,
            'pool_type': pool_type,
            'cls_type': cls_type,
            'scale': scale,
            'margin': margin,
            'with_bnneck': with_bnneck,
            'norm_type': norm_type,
            'cls_enable': cls_enable,
            'loss_names': loss_names,
            'mixstyle': mixstyle
        }

    def forward(self, backbone_outputs, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        if isinstance(backbone_outputs, dict):
            features = backbone_outputs['features']
            targets = backbone_outputs['targets']
        else:
            features = backbone_outputs
        
        pool_feat = self.neck_bn(self.pool_layer(features))
        neck_feat = self.bottleneck(pool_feat.view(pool_feat.shape[0],-1))
        #neck_feat = neck_feat[..., 0, 0]

        # Evaluation
        # fmt: off
        if not self.training: return neck_feat
        # fmt: on

        # Training

        # fmt: off
        if self.neck_feat == 'before':  feat = pool_feat[..., 0, 0]
        elif self.neck_feat == 'after': feat = neck_feat
        else:                           raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on
        
        # Pass logits.clone() into cls_layer, because there is in-place operations
        if self.cls_enable:
            if self.cls_layer.__class__.__name__ == 'Linear':
                logits = F.linear(neck_feat, self.weight)
            else:
                logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))
            
            if self.mixstyle:
                logits_0, logits_1 = logits.chunk(2,0)
                targets_0, targets_1 = targets.chunk(2)
                cls_outputs = self.cls_layer(logits_0.clone(), targets_0)
                pred_class_logits = logits_0.mul(self.cls_layer.s)
            else:
                cls_outputs = self.cls_layer(logits.clone(), targets)
                pred_class_logits = logits.mul(self.cls_layer.s)


            return {
                "cls_outputs": cls_outputs,
                "pred_class_logits": pred_class_logits,
                "features": feat,
                "weight": self.weight}

        else:
            #print('only one output!!!!!!!!') 
            return {"features": feat,}
