import torch
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import *
from siamese import Encoder, Predictor
from stn import stn_net, STNModule
from utils import AverageMeter

def trainer(config):
    
    print('Loading Datasets')
    train_dataset = FSAD_Dataset_train(
        config = config,
        is_train=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 1,
        shuffle = True,
        num_workers=config.model.num_workers
    )

    print('Loading Models')
    device = config.model.device
    STN = stn_net(config.model).to(device)
    ENC = Encoder().to(device)
    PRED = Predictor().to(device)  

    STN_optimizer = optim.SGD(STN.parameters(), lr=config.model.lr, momentum=config.model.momentum)
    ENC_optimizer = optim.SGD(ENC.parameters(), lr=config.model.lr, momentum=config.model.momentum)
    PRED_optimizer = optim.SGD(PRED.parameters(), lr=config.model.lr, momentum=config.model.momentum)

    models = [STN, ENC, PRED]
    optimizers = [STN_optimizer, ENC_optimizer, PRED_optimizer]
    init_lrs = [config.model.lr, config.model.lr, config.model.lr]


    print('Start Training……')
    for epoch in range(1, config.model.epochs):
        adjust_learning_rate(optimizers, init_lrs, epoch, config.model.epochs)
        STN = models[0]
        ENC = models[1]
        PRED = models[2]

        STN_optimizer = optimizers[0]
        ENC_optimizer = optimizers[1]
        PRED_optimizer = optimizers[2]

        STN.train()
        ENC.train()
        PRED.train()

        total_losses = AverageMeter()
        for (query_img, support_img_list, _) in tqdm(train_loader):
            STN_optimizer.zero_grad()
            ENC_optimizer.zero_grad()
            PRED_optimizer.zero_grad()
            
            query_img = query_img.squeeze(0).to(device) # [32, 3, 224, 224]
            query_feat = STN(query_img)  # [32, 256, 14, 14]
            print(query_feat.shape)
            return 
            support_img = support_img_list.squeeze(0).to(device) # [32, 2, 3, 224, 224]
            B,K,C,H,W = support_img.shape

            support_img = support_img.view(B * K, C, H, W)
            support_feat = STN(support_img)
            support_feat = support_feat / K

            _, C, H, W = support_feat.shape
            support_feat = support_feat.view(B, K, C, H, W)
            support_feat = torch.sum(support_feat, dim=1)
            print(support_img_list.shape)
            return


def adjust_learning_rate(optimizers, init_lrs, epoch, epochs):
    """Decay the learning rate based on schedule"""
    for i in range(3):
        cur_lr = init_lrs[i] * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
        for param_group in optimizers[i].param_groups:
            param_group['lr'] = cur_lr


