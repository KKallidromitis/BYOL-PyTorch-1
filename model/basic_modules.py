#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models
from utils.mask_utils import sample_masks
import torch.nn.functional as F
from model.models import MLP,Masknet,FPN

class EncoderwithProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mask_rois = config['loss']['mask_rois']
        self.pool_size = config['loss']['pool_size']
        self.train_batch_size = config['data']['train_batch_size']
        
        if config['log']['wandb_enable']:
            from utils.visualize_masks import wandb_sample
            
        # backbone
        pretrained = config['model']['backbone']['pretrained']
        net_name = config['model']['backbone']['type']
        base_encoder = models.__dict__[net_name](pretrained=pretrained)
        
        if self.pool_size==7:
            self.encoder = nn.Sequential(*list(base_encoder.children())[:-2])
        else:
            self.C1 = nn.Sequential(*list(base_encoder.children())[:4])
            self.C2 = nn.Sequential(*list(base_encoder.children())[4])
            self.C3 = nn.Sequential(*list(base_encoder.children())[5])
            self.C4 = nn.Sequential(*list(base_encoder.children())[6])
            self.C5 = nn.Sequential(*list(base_encoder.children())[7])
            self.fpn = FPN(2048,self.pool_size)
            
        # projection
        input_dim = config['model']['projection']['input_dim']
        hidden_dim = config['model']['projection']['hidden_dim']
        output_dim = config['model']['projection']['output_dim']
        self.projetion = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,mask_roi=self.mask_rois)        
        
    def forward(self, x, masks, mnet=None,wandb_id=None,net_type=None):
        #import ipdb;ipdb.set_trace()
        if self.pool_size==7:
            x = self.encoder(x) #(B, 2048, pool_size, pool_size)
        else:
            x = self.C1(x)
            x = self.C2(x)
            c2_out = x
            x = self.C3(x)
            c3_out = x
            x = self.C4(x)
            c4_out = x
            x = self.C5(x)     
            x = self.fpn(x,c2_out,c3_out,c4_out)
            
        masks,mask_ids = sample_masks(masks,self.mask_rois)

        # Wandb Logging
        if wandb_id!=None:
            wandb_sample(self.mask_rois,self.pool_size,masks[wandb_id],masks[wandb_id+self.train_batch_size],'sample_masks_'+net_type)
        
        if mnet!= None:
            masks = mnet(x.detach(),masks.to('cuda'))
        
        # Wandb Logging
        if wandb_id!=None:
            wandb_sample(self.mask_rois,self.pool_size,masks[wandb_id],masks[wandb_id+self.train_batch_size],'masknet_masks_'+net_type)
        
        # Detcon mask multiply
        bs, emb, emb_x, emb_y  = x.shape
        x = x.permute(0,2,3,1) #(B, pool_size, pool_size, 2048)
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
