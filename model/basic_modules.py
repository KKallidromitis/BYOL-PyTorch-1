#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models
from utils.mask_utils import sample_masks
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,mask_roi=16):
        super().__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(mask_roi)
        self.relu1 = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.l2(x)
        return x
    
class Masknet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2048, 2048, 1)
        self.conv2 = nn.Conv2d(2048, 2048, 1)
        self.conv3 = nn.Conv2d(2048, 16, 1)
        self.norm = nn.BatchNorm2d(16, 16)

    def forward(self, x):
        y = self.norm(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x))))))
        return y

class EncoderwithProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        # backbone
        pretrained = config['model']['backbone']['pretrained']
        net_name = config['model']['backbone']['type']
        base_encoder = models.__dict__[net_name](pretrained=pretrained)
        self.encoder = nn.Sequential(*list(base_encoder.children())[:-2])

        # projection
        input_dim = config['model']['projection']['input_dim']
        hidden_dim = config['model']['projection']['hidden_dim']
        output_dim = config['model']['projection']['output_dim']
        self.projetion = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)        
        
    def forward(self, x, masks, mnet=None):
        #import ipdb;ipdb.set_trace()
        x = self.encoder(x) #(B, 2048, 7, 7)
        masks,mask_ids = sample_masks(masks)
        
        if mnet!=None:
            pertubation = torch.reshape(mnet(x.detach()),(-1, 16, 49))
            masks = pertubation + masks.to('cuda')
        
        
        # Detcon mask multiply
        bs, emb, emb_x, emb_y  = x.shape
        masks_area = masks.sum(axis=-1, keepdims=True)
        smpl_masks = masks / torch.maximum(masks_area, torch.ones_like(masks_area))
        embedding_local = torch.reshape(x,[bs, emb_x*emb_y, emb])
        x = torch.matmul(smpl_masks.float().to('cuda'), embedding_local)
        
        x = self.projetion(x)
        return x, mask_ids

class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        # predictor
        input_dim = config['model']['predictor']['input_dim']
        hidden_dim = config['model']['predictor']['hidden_dim']
        output_dim = config['model']['predictor']['output_dim']
        self.predictor = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, x, mask_ids):
        return self.predictor(x), mask_ids
