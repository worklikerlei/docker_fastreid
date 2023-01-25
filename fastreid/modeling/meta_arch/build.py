# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import copy

from fastreid.utils.registry import Registry

import logging
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.
The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)

    if cfg.MODEL.PRETRAIN:
        print('Loading pretrained model.........................')
        state_dict = torch.load(cfg.MODEL.PRETRAIN_PATH, map_location=torch.device('cpu'))
       
        if 'model' in state_dict:
            print('model loaded..........')
            state_dict = state_dict['model']
        else:
            print('No model loaded')
        '''
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}#filter out unnecessary keys 
        model_dict.update(pretrained_dict)
        '''
        
        import collections
        renamed_dict = collections.OrderedDict()
        # print(state_dict.items())
        for key,value in state_dict.items():
            #print(key)
            #input()
            if 'encoder_q.0.' in key:
                new_key = 'backbone.' +  key.split('encoder_q.0.')[1]
                renamed_dict[new_key] = value
            else:
                renamed_dict[key] = value
                
        model_dict = copy.deepcopy(renamed_dict)
        for item in state_dict.keys():
            if 'head' in item:
                model_dict.pop(item)
        # print(model_dict.keys())
        
        logger = logging.getLogger(__name__)
        
        incompatible = model.load_state_dict(model_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )
        print('Pretrained model successfully loaded!!!!!!!')
    
    model.to(torch.device(cfg.MODEL.DEVICE))
    total = sum([param.nelement() for param in model.parameters()])
    print(" Number of parameter: %.2fM" % (total/1e6))
    return model
