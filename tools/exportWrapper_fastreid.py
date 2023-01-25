import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pdb
import sys
sys.path.append('.')
import os

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_setup


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--output",
        default='./output/',
        help='path to save features'
    )
    parser.add_argument(
        "--weights",
        help="path to load weights",
        default='./logs/veri/sbs_R50-ibn/model_best.pth',
    )
    parser.add_argument(
        "--config-file", 
        default="./configs/VeRi/sbs_R50-ibn_finetune.yml", 
        metavar="FILE", 
        help="path to config file")
    return parser

    
   
if __name__ == '__main__':
   
    img = torch.zeros((1,3,256,256))
    args = get_parser().parse_args()
    
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    default_setup(cfg, args)
  

    class ReIdWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # build model
            cfg.defrost()
            cfg.MODEL.BACKBONE.PRETRAIN = False
            b_model = DefaultTrainer.build_model(cfg)
            # arch = 'resnet50'
            # model_fc = models.__dict__[arch]()
            self.model = b_model.cpu()
            # remove fc layer
            # self.model = torch.nn.Sequential(*(list(b_model.children())[:-1]))
            
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
            # print(self.model)
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
    features = model(img)  # dry run


    print("type(y):", type(features))
    print("shape(y):", features.shape)
    

    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        f = args.weights.split('/')[-1].replace('.pth', '.torchscript.wrapper.pt')  # filename
        ts = torch.jit.trace(model, img)
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        ts.save(args.output+'/'+f)
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)
        
        
    # test_model = torch.jit.load(args.output+'/'+f)
        
    # feat = test_model(img)
    
    # a = features - feat
    
    # test = 1
