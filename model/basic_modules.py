#-*- coding:utf-8 -*-
import imp
import torch
import torch.nn as nn
from torchvision import models
from utils.mask_utils import sample_masks
import torch.nn.functional as F
import numpy as np
import timm
from .vit_deconv import vit_base_patch16 as vit_base_patch16_deconv
from torchvision.models._utils import IntermediateLayerGetter

# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim,mask_roi=16):
#         super().__init__()

#         self.l1 = nn.Linear(input_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.l2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = self.l1(x)
#         batch_size,n_channal,n_emb= x.shape # B * 16 * f
#         x = self.bn1(x.reshape(batch_size*n_channal,n_emb))
#         x = x.reshape(batch_size,n_channal,n_emb)
#         x = self.relu1(x)
#         x = self.l2(x)
#         return x
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,mask_roi=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        if hidden_dim > 0:
            self.l1 = nn.Conv2d(input_dim, hidden_dim,1)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            self.relu1 = nn.ReLU(inplace=True)
            self.l2 = nn.Conv2d(hidden_dim, output_dim,1)
        else:
            self.l1 = nn.Conv2d(input_dim, output_dim,1)
        

    def forward(self, x):
        x = self.l1(x)
        if self.hidden_dim > 0:
            x = self.bn1(x)
            #x = x.reshape(batch_size,n_channal,n_emb)
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

class VitWrapper(nn.Module):
    def __init__(self,backbone,feature_resolution=14,deconv=False):
        super().__init__()
        self.backbone = backbone
        self.dim = feature_resolution
        self.deconv = deconv
    @property
    def forward_features(self):
        return self.backbone.forward_features

    def forward(self,x):
        if self.deconv:
            return self.forward_features(x)[-1] # c5 7x7
        else: # standrad vit
            feat = self.backbone.forward_features(x).permute(0,2,1)[:,:,1:]
            n,c,hw = feat.shape
            return feat.reshape(n,c,self.dim,self.dim) # B X C X H X W

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
    input_transform=Nons=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activate): ReLe, ignore_index=255, align_corners=False
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
      (bn): SyncBatchNorm(256, epU(inplace=True)
    )
  )
    '''
    def __init__(self,attention=False,upsample=56):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.conv2 =  nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.convSeg = nn.Conv2d(256, 16, kernel_size=(1, 1), stride=(1, 1))
        self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        #self.softmax = nn.Softmax2d()
        self.finalbn = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.attention = attention
        if self.attention:
            self.atten = SpatialAttentionBlock(256,64,256)
            self.conv3 = nn.Conv2d(256, 256, 1, bias=False)
            
        self.upsample = upsample
        
    def forward(self, x):
        #7x7x2048
        #TODO: RESAMPLE ATTENTION
        upsample = self.upsample
        if self.attention:
            x_norm = F.interpolate(x, (upsample,upsample), mode='bilinear',align_corners=False)
            x_norm = F.normalize(x,dim=1)
            atten =torch.einsum('bcxy,bcij->bxyij',x_norm,x_norm) #B X H X W X H X W
        x = self.conv1(x)
        x = F.interpolate(x, (upsample,upsample), mode='bilinear',align_corners=False)
        x = self.conv2(x)
        x = self.dropout(x)
        # 
        if self.attention:
            #x = self.atten(x)+self.conv3(x)
            x = torch.einsum('bxyij,bdij->bxyd',atten,x) +self.conv3(x)# B X( H X W) X d_emb, # Similar matrice will have simlar embeding
        x = self.convSeg(x)
        x = self.finalbn(x)
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
        if net_name == 'vit':
            base_encoder = timm.create_model('vit_base_patch16_224', pretrained=False,global_pool='',class_token =True)
            self.encoder = VitWrapper(base_encoder,config['model']['backbone']['feature_resolution'])
        elif net_name == 'vit-deconv':
            base_encoder = vit_base_patch16_deconv()
            self.encoder = VitWrapper(base_encoder,config['model']['backbone']['feature_resolution'],deconv=True)
        else:
            ## resnet
            base_encoder = models.__dict__[net_name](pretrained=pretrained)
            base_encoder = nn.Sequential(*list(base_encoder.children())[:-2])
            self.encoder = IntermediateLayerGetter(base_encoder, return_layers={'7':'c5','6':'c4','5':'c3'})



        self.projections = nn.ModuleDict({
            config['model']['projection']['keys'][k]: MLP(
            input_dim=config['model']['projection']['input_dim'][k],
            hidden_dim=config['model']['projection']['hidden_dim'][k],
            output_dim=config['model']['projection']['output_dim'],
            ) 
            for k in range(len(config['model']['projection']['keys']))})
        
        self.projection_map = config['model']['projection_map']
            
        
    def forward(self, x,mode="features",key=None):
        #import ipdb;ipdb.set_trace()
        if mode == 'projection':
            if key:
                layer = self.projections[self.projection_map[key]]
                return layer(x)
            else:
                for k,v in x.items():
                    layer = self.projections[self.projection_map[k]]
                    out_dict[k] = layer(v) # layer(x[k])
        else:
            x = self.encoder(x) #(B, 2048, 7, 7) # 512, 1024, 2048
            # dict 
            out_dict = {}
            for k,layer in self.projections.items():
                out_dict[k] = x[k] # layer(x[k])

            return out_dict

class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        # predictor
        input_dim = config['model']['predictor']['input_dim']
        hidden_dim = config['model']['predictor']['hidden_dim']
        output_dim = config['model']['predictor']['output_dim']
        self.predictor = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, x):
        return self.predictor(x)


class FCNMaskNetV2(nn.Module):
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
    def __init__(self,attention=False,upsample=14):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.conv2 =  nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.convSeg = nn.Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
        self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        #self.softmax = nn.Softmax2d()
        self.finalbn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.attention = attention
        if self.attention:
            self.atten = SpatialAttentionBlock(256,64,256)
            self.conv3 = nn.Conv2d(256, 256, 1, bias=False)
            
        self.upsample = upsample
        
    def forward(self, x):
        #7x7x2048
        #TODO: RESAMPLE ATTENTION
        upsample = self.upsample
        if self.attention:
            x_norm = F.interpolate(x, (upsample,upsample), mode='bilinear',align_corners=False)
            x_norm = F.normalize(x_norm,dim=1)
            atten =torch.einsum('bcxy,bcij->bxyij',x_norm,x_norm) #B X H X W X H X W
        x = self.conv1(x)
        x = F.interpolate(x, (upsample,upsample), mode='bilinear',align_corners=False)
        x = self.conv2(x)
        x = self.dropout(x)
        # 
        if self.attention:
            #x = self.atten(x)+self.conv3(x)
            #print(x.shape,atten.shape,self.conv3(x).shape)
            x = torch.einsum('bxyij,bdij->bxyd',atten,x).permute(0,3,1,2) #+self.conv3(x)# B X( H X W) X d_emb, # Similar matrice will have simlar embeding
        x = self.finalbn(x)
        x = self.convSeg(x)
        #x = self.finalbn(x)
        
        #x = self.softmax(x)
        #breakpoint()
        # x = self.atten(x) +self.conv3(x)
        # x = F.softmax(x,dim=1) # B X C X H X W
        return x