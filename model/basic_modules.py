#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_type="linear"):
        super().__init__()
        
        if layer_type == "linear":
            self.l1 = nn.Linear(input_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, output_dim)
        elif layer_type == "conv":
            self.l1 = nn.Conv2d(input_dim, hidden_dim, (1,1), stride=1, padding=0, bias=True)
            self.l2 = nn.Conv2d(hidden_dim, output_dim, (1,1), stride=1, padding=0, bias=True)
        else:
            raise NotImplementedError()


        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

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

        # projection
        input_dim = config['model']['projection']['input_dim']
        hidden_dim = config['model']['projection']['hidden_dim']
        output_dim = config['model']['projection']['output_dim']
        self.projection_type = config["model"]["projection"]["layer"]
        self.projection = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, layer_type=self.projection_type)

    def forward(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        if self.projection_type == "linear": #Final output will be (B, 256), o.w. (B, 256, 1, 1)
            x = torch.flatten(x, 1)
        x = self.projection(x)
        return x

class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        # predictor
        input_dim = config['model']['predictor']['input_dim']
        hidden_dim = config['model']['predictor']['hidden_dim']
        output_dim = config['model']['predictor']['output_dim']
        self.predictor = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, x):
        if x.ndim == 4: #(B, C, 1, 1)
            x = torch.flatten(x, 1) # (B, C)
        return self.predictor(x)
