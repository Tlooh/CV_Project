import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

import torch
import numpy as np
import argparse
from omegaconf import OmegaConf
from train import trainer





def parse_args():
    parser = argparse.ArgumentParser('DDAD')    
    parser.add_argument('-cfg', '--config', 
                                default= os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.yaml'), 
                                help='config file')
    parser.add_argument('--train', 
                                default= True, 
                                help='Train the diffusion model')
    parser.add_argument('--seed', type=int, default=42, help='manual seed')

    parser.add_argument('--detection', 
                                default= False, 
                                help='Detection anomalies')
    parser.add_argument('--domain_adaptation', 
                                default= False, 
                                help='Domain adaptation')
    parser.add_argument('--stn_mode', type=str, default='rotation_scale',
                        help='[affine, translation, rotation, scale, shear, rotation_scale, translation_scale, rotation_translation, rotation_translation_scale]')
    args, unknowns = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if args.train:
        print('Training...')
        trainer(config)





