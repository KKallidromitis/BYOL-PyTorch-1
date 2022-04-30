#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.l2(x)
        return x

class EncoderwithProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        # backbone
        pretrained = config['model']['backbone']['pretrained']
        net_name = config['model']['backbone']['type']
        base_encoder = models.__dict__[net_name](pretrained=pretrained)
        self.encoder = nn.Sequential(*list(base_encoder.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # projection
        input_dim = config['model']['projection']['input_dim']
        hidden_dim = config['model']['projection']['hidden_dim']
        output_dim = config['model']['projection']['output_dim']
        self.projetion = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, x):
        import ipdb; ipdb.set_trace()
        encoding = self.encoder(x) #(B, 2048, 7, 7)
        x = self.pool(encoding) #(B, 2048, 1, 1)
        x = torch.flatten(x, 1) #(B, 2048)
        x = self.projetion(x) #(B, 256)
        return encoding, x

class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        # predictor
        input_dim = config['model']['predictor']['input_dim']
        hidden_dim = config['model']['predictor']['hidden_dim']
        output_dim = config['model']['predictor']['output_dim']
        self.predictor = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)


def cosine_attention(pixels,ref_vec):
    '''
    ref_vec: B X dim_emb
    pixels: B X dim_emb X H X W X 
    '''
    ref_vec = F.normalize(ref_vec,dim=-1)
    pixels = F.normalize(pixels,dim=-1)
    atten = torch.einsum('bcxy,bc->bxy',pixels,ref_vec)
    return atten