#-*- coding:utf-8 -*-
import torch
import math
import torch.nn as nn
from torchvision import models
#from utils.mask_utils import sample_masks
import torch.nn.functional as F
from utils.visualize_masks import wandb_sample

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,mask_roi=16):
        super().__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) #TODO: CHECK MASK_ROI????
        self.relu1 = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.mask_roi = mask_roi

    def forward(self, x):
        # x is (B, mask_roi (or 7x7=49 if passing in all of c5), 2048)
        # Else, x is (B, Area (e.g. 7x7), 2048)
        b = x.size(0)
        num_embedings = x.size(1)
        x = self.l1(x)
        x = torch.reshape(x, (b*num_embedings, -1)) #(B*16, 4096)
        x = self.bn1(x)
        x = torch.reshape(x, (b, num_embedings, -1)) #(B, 16, 4096)
        x = self.relu1(x)
        x = self.l2(x)
        return x
    
class Masknet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mask_rois = config['loss']['mask_rois']
        self.pool_size = config['loss']['pool_size']
        
        self.input_channels = config['model']['projection']['output_dim']


        # FCN Head
        # self.mask_head = nn.Sequential(
        #                             nn.Conv2d(self.input_channels, self.hidden_channels, 3, padding=1, bias=False),
        #                             nn.SyncBatchNorm(self.hidden_channels),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(self.hidden_channels, self.mask_rois, 1),
        #                             )
        self.mask_head = nn.Sequential(
                                        nn.Conv2d(self.input_channels, self.mask_rois, 1)
                                        )

    def forward(self, x):
        # Comment shapes assume input is (B, 256, h, w)
        b = x.size(0)
        x = x.detach() #Detaching just in case #TODO: Check Detach/No-Detach
        mask = self.mask_head(x) #(B, 256, h, w)
        return mask, torch.arange(0, self.mask_rois).expand(b*2, -1).cuda()



    
class EncoderwithProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mask_rois = config['loss']['mask_rois']
        self.pool_size = config['loss']['pool_size']
        self.train_batch_size = config['data']['train_batch_size']

        # backbone
        pretrained = config['model']['backbone']['pretrained']
        net_name = config['model']['backbone']['type']
        base_encoder = models.__dict__[net_name](pretrained=pretrained)
        
        if self.pool_size==7:
            self.encoder = nn.Sequential(*list(base_encoder.children())[:-2])
        # else:
        #     self.C1 = nn.Sequential(*list(base_encoder.children())[:4])
        #     self.C2 = nn.Sequential(*list(base_encoder.children())[4])
        #     self.C3 = nn.Sequential(*list(base_encoder.children())[5])
        #     self.C4 = nn.Sequential(*list(base_encoder.children())[6])
        #     self.C5 = nn.Sequential(*list(base_encoder.children())[7])
        #     self.fpn = FPN(2048,self.pool_size)
            
        # projection
        input_dim = config['model']['projection']['input_dim']
        hidden_dim = config['model']['projection']['hidden_dim']
        output_dim = config['model']['projection']['output_dim']
        self.projetion = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,mask_roi=self.mask_rois)

    def forward(self, x, masks, mask_ids, is_embeddings=False, wandb_id=None):
        # is_embeddings, if x is already encodings
        if not is_embeddings and self.pool_size == 7:
            x = self.encoder(x) #(B, 2048, pool_size, pool_size)

        # Detcon mask multiply
        bs, emb, emb_x, emb_y  = x.shape
        x = x.permute(0,2,3,1) #(B, pool_size, pool_size, 2048)
        x = torch.reshape(x, [bs, emb_x*emb_y, emb]) #(B, pool_size**2, 2048)
        masks = torch.reshape(masks, [bs, self.mask_rois, -1]) #(B, 16, pool_size, pool_size) -> (B, 16, pool_size**2)
        x = torch.einsum('bma, bae -> bme', masks, x) #(B, 16, 2048)
        # TODO: CHECK RESCALING (BELOW)
        # masks_area = masks.sum(axis=-1, keepdims=True)
        # smpl_masks = masks / torch.maximum(masks_area, torch.ones_like(masks_area))
        # embedding_local = torch.reshape(x,[bs, emb_x*emb_y, emb])
        # x = torch.matmul(smpl_masks.float().to('cuda'), embedding_local)
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

    def forward(self, x, mask_ids=[]):
        output = self.predictor(x)
        if len(mask_ids) > 0:
            return output, mask_ids
        return self.predictor(x)
