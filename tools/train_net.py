#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys
import os
#import os.path as osp
#project_dir = osp.dirname(osp.dirname(sys.argv[0]))

import torch
import torch.nn.functional as F

sys.path.append('.')
sys.path.append('..')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = DefaultTrainer.test(cfg, model)
        return res

    if args.save_only:

        class ReIdWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # build model
                cfg.defrost()
                cfg.MODEL.BACKBONE.PRETRAIN = False
                b_model = DefaultTrainer.build_model(cfg)
                # arch = 'resnet50'
                # model_fc = models.__dict__[arch]()
                # remove fc layer
                # self.model = torch.nn.Sequential(*(list(model_fc.children())[:-1]))
                self.model = b_model.cpu()
                # load weights
                # Checkpointer(self.model).load(args.weights) 
                checkpoint = torch.load(args.weights)
                state_dict = checkpoint['model'] 
                # remove the momentum encoder weights
                # for k in list(state_dict.keys()):
                #     if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                #         state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                #     del state_dict[k]
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()


            def postprocess(self, features):
                # features = features.squeeze(2).squeeze(2)
                features = F.normalize(features)
                return features

            def forward(self, img):
                # forward
                features = self.model(img)
                features = self.postprocess(features)
                return features

        model = ReIdWrapper()

        inputs = torch.randn(1,3,cfg.INPUT.SIZE_TEST[0],cfg.INPUT.SIZE_TEST[1])
        features = model(inputs)  # dry run


        print("type(y):", type(features))
        print("shape(y):", features.shape)


        try:
            print('\nStarting TorchScript export with torch %s...' % torch.__version__)
            f = args.weights.split('/')[-1].replace('.pth', '.torchscript.wrapper.pt')  # filename
            ts = torch.jit.script(model, inputs)
            if not os.path.exists(args.output):
                os.makedirs(args.output)
            ts.save(args.output+'/'+f)
            print('TorchScript export success, saved as %s' % f)
        except Exception as e:
            print('TorchScript export failure: %s' % e)

        return 1


    print('-----------------------------------------------------------------')
    print(cfg.MODEL.BACKBONE.PRETRAIN)
    print('-----------------------------------------------------------------')
    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
