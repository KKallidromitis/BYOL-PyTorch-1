#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models
from utils.mask_utils import sample_masks
import torch.nn.functional as F
from utils.visualize_masks import wandb_sample

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
    def __init__(self, config):
        super().__init__()
        self.mask_rois = config['loss']['mask_rois']
        self.conv1 = nn.Conv2d(2048, 2048, 1)
        self.conv2 = nn.Conv2d(2048, 2048, 1)
        self.conv3 = nn.Conv2d(2048, self.mask_rois, 1)
        self.softmax = nn.Softmax(dim=-1)
        #self.norm = nn.BatchNorm2d(self.mask_rois, self.mask_rois)

    def forward(self, x, masks):
        #import ipdb;ipdb.set_trace()
        x = self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = torch.reshape(x,(-1, self.mask_rois, 49))
        y = x+masks
        y = self.softmax(y)
        return y

class EncoderwithProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mask_rois = config['loss']['mask_rois']
        self.train_batch_size = config['data']['train_batch_size']
        # backbone
        pretrained = config['model']['backbone']['pretrained']
        net_name = config['model']['backbone']['type']
        base_encoder = models.__dict__[net_name](pretrained=pretrained)
        self.encoder = nn.Sequential(*list(base_encoder.children())[:-2])

        # projection
        input_dim = config['model']['projection']['input_dim']
        hidden_dim = config['model']['projection']['hidden_dim']
        output_dim = config['model']['projection']['output_dim']
        self.projetion = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,mask_roi=self.mask_rois)        
        
    def forward(self, x, masks, mnet=None,wandb_id=None,net_type=None):
        #import ipdb;ipdb.set_trace()
        x = self.encoder(x) #(B, 2048, 7, 7)
        masks,mask_ids = sample_masks(masks,self.mask_rois)

        if wandb_id!=None:
            wandb_sample(torch.reshape(masks[wandb_id],(self.mask_rois,7,7)).detach().cpu().numpy(),
                         torch.reshape(masks[wandb_id+self.train_batch_size],(self.mask_rois,7,7)).detach().cpu().numpy(),
                         'sample_masks_'+net_type)
        
        if mnet!= None:
            masks = mnet(x.detach(),masks.to('cuda'))
            
        if wandb_id!=None:
            wandb_sample(torch.reshape(masks[wandb_id],(self.mask_rois,7,7)).detach().cpu().numpy(),
                 torch.reshape(masks[wandb_id+self.train_batch_size],(self.mask_rois,7,7)).detach().cpu().numpy(),
                         'masknet_masks_'+net_type)
        
        
        # Detcon mask multiply
        bs, emb, emb_x, emb_y  = x.shape
        x = x.permute(0,2,3,1) #(B, 7, 7, 2048)
        masks_area = masks.sum(axis=-1, keepdims=True)
        smpl_masks = masks / torch.maximum(masks_area, torch.ones_like(masks_area))
        embedding_local = torch.reshape(x,[bs, emb_x*emb_y, emb])
        x = torch.matmul(smpl_masks.float().to('cuda'), embedding_local)
        
        x = self.projetion(x)
        return x, mask_ids

class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mask_rois = config['loss']['mask_rois']
        # predictor
        input_dim = config['model']['predictor']['input_dim']
        hidden_dim = config['model']['predictor']['hidden_dim']
        output_dim = config['model']['predictor']['output_dim']
        self.predictor = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,mask_roi=self.mask_rois)

    def forward(self, x, mask_ids):
        return self.predictor(x), mask_ids
