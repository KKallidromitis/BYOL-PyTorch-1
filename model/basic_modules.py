#-*- coding:utf-8 -*-
import imp
import torch
import torch.nn as nn
from torchvision import models
from utils.mask_utils import sample_masks
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,mask_roi=16):
        super().__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.l1(x)
        batch_size,n_channal,n_emb= x.shape # B * 16 * f
        x = self.bn1(x.reshape(batch_size*n_channal,n_emb))
        x = x.reshape(batch_size,n_channal,n_emb)
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

class SpatialAttentionBlock(nn.Module):

    def __init__(self,in_dim=2048,atten_dim=64,emb_dim=64):
        super().__init__()
        self.k_map = nn.Conv2d(in_dim,atten_dim,1)
        self.q_map = nn.Conv2d(in_dim,atten_dim,1)
        self.v_map = nn.Conv2d(in_dim,emb_dim,1)
        self.atten_dim = atten_dim

    def forward(self, x):
        #B X C X (H X W)
        b,_,h,w = x.shape
        q = self.q_map(x)   # B X atten_dim X (H X W) 
        k = self.k_map(x)   # B X atten_dim X (H X W) 
        v = self.v_map(x)   # B X emb_dim X (H X W) 
        atten =torch.einsum('bcxy,bcij->bxyij',q,k) # B X (H X W) X( H X W)
        atten /= self.atten_dim
        atten = atten.reshape(b,h,w,h*w)
        atten = F.softmax(atten,dim=-1)
        atten = atten.reshape(b,h,w,h,w)
        out = torch.einsum('bxyij,bdij->bxyd',atten,v) # B X( H X W) X d_emb
        return out.permute(0,3,1,2)
        ##torch.bmm(k.reshape(2,64,49).permute(0,2,1),q.reshape(2,64,49)
        

class SpatialAttentionMasknet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2048, 2048, 1)
        self.conv2 = nn.Conv2d(2048, 2048, 1)
        self.atten = SpatialAttentionBlock(2048,64,16)
        self.conv3 = nn.Conv2d(2048, 16, 1)

    def forward(self, x):
        #7x7x2048
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        #breakpoint()
        x = self.atten(x) +self.conv3(x)
        x = F.softmax(x,dim=1) # B X C X H X W
        return x


class FCNMaskNet(nn.Module):
    '''
      (decode_head): FCNHead(
    input_transform=None, ignore_index=255, align_corners=False
    (loss_decode): CrossEntropyLoss()
    (conv_seg): Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))
    (dropout): Dropout2d(p=0.1, inplace=False)
    (convs): Sequential(
      (0): ConvModule(
        (conv): Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6), bias=False)
        (bn): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU(inplace=True)
      )
      (1): ConvModule(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6), bias=False)
        (bn): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU(inplace=True)
      )
    )
    (conv_cat): ConvModule(
      (conv): Conv2d(2304, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activate): ReLU(inplace=True)
    )
  )
    '''
    def __init__(self,attention=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6), bias=False),
            nn.SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.conv2 =  nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6), bias=False),
            nn.SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.convSeg = nn.Conv2d(256, 16, kernel_size=(1, 1), stride=(1, 1))
        self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        #self.softmax = nn.Softmax2d()
        self.attention = attention
        if self.attention:
            self.atten = SpatialAttentionBlock(256,64,256)
            self.conv3 = nn.Conv2d(256, 256, 1)
        
    def forward(self, x):
        #7x7x2048
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        # 
        if self.attention:
            x = self.atten(x)+self.conv3(x)
        x = self.convSeg(x)
        #x = self.softmax(x)
        #breakpoint()
        # x = self.atten(x) +self.conv3(x)
        # x = F.softmax(x,dim=1) # B X C X H X W
        return x
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
        
    def forward(self, x, masks,mask_ids, mnet=None):
        #import ipdb;ipdb.set_trace()
        x = self.encoder(x) #(B, 2048, 7, 7)
        #breakpoint()
        # Detcon mask multiply
        # if mnet!= None:
        #     masks = torch.reshape(mnet(x),(-1, 16, 49))

        ## Passed in mask, direct pooling
        bs, emb, emb_x, emb_y  = x.shape
        x = x.permute(0,2,3,1) # (B,7,7,2048)

        #breakpoint()
        masks_area = masks.sum(axis=-1, keepdims=True)
        smpl_masks = masks / torch.maximum(masks_area, torch.ones_like(masks_area))
        #smpl_masks = torch.ones((bs,1,49))
        #Overwreite with standard BYOL
        #breakpoint()
        embedding_local = torch.reshape(x,[bs, emb_x*emb_y, emb])
        #breakpoint()
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
