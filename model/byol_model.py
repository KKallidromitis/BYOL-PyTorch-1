#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_modules import EncoderwithProjection, Predictor

class BYOLModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # online network
        self.online_network = EncoderwithProjection(config)

        # target network
        self.target_network = EncoderwithProjection(config)

        # predictor
        self.predictor = Predictor(config)

        self._initializes_target_network()

    @torch.no_grad()
    def _initializes_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient

    @torch.no_grad()
    def _update_target_network(self, mm):
        """Momentum update of target network"""
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.mul_(mm).add_(1. - mm, param_q.data)

    def forward(self, view1, view2, view0, mm):

        b = view1.size(0)

        # Get online c5 outputs
        online_c5_output = self.online_network.encoder(torch.cat([view1, view2, view0])) #(3B, 2048, 7, 7) in (view1, view2, view0) order in batch dim
        online_v1, online_v2, online_v0 = online_c5_output[:b], online_c5_output[b:2*b], online_c5_output[2*b:]
        h, w = online_v1.shape[-2:] #C5 output is always same (b/c view1, view2 shape is same)

        # Compute similarity
        if torch.linalg.norm(online_v0).item() < 1:
            print(online_v0, torch.linalg.norm(online_v0))

        online_v0_normalized = F.normalize(online_v0, dim=1).clone().detach()
        online_v1_normalized = F.normalize(online_v1, dim=1)
        online_v2_normalized = F.normalize(online_v2, dim=1)
        atten_ab = torch.einsum("bchw, bcij -> bhwij", online_v0_normalized, online_v1_normalized) # (B, 7,7, 7,7)
        assert not torch.all(torch.isnan(atten_ab))
        atten_ba = torch.einsum("bchw, bcij -> bhwij", online_v0_normalized, online_v2_normalized)
        assert not torch.all(torch.isnan(atten_ab))

        #Relu attention
        atten_ab = F.relu(atten_ab.reshape((b, h, w, -1))) #(B, 7, 7, 49)
        atten_ba = F.relu(atten_ba.reshape(b, h, w, -1)) #(B, 7, 7, 49)
        atten_ab = atten_ab.reshape((b, h, w, h, w)) #(B, 7, 7, 7, 7)
        atten_ba = atten_ba.reshape_as(atten_ab)

        # Get values
        online_projections = self.online_network.projection(torch.cat([online_v1, online_v2])) #(2B, 256, 7, 7) in (view1, view2) order
        proj_v1, proj_v2 = online_projections[:b], online_projections[b:] #(B, 256, 7, 7)

        # Get attention weighted values
        g1 = torch.einsum("bhwij, bcij -> bchw", atten_ab, proj_v1) #(B, 256, 7, 7)
        g2 = torch.einsum("bhwij, bcij -> bchw", atten_ba, proj_v2)
        g1 = F.adaptive_avg_pool2d(g1, (1,1))
        g2 = F.adaptive_avg_pool2d(g2, (1,1))

        # Get predictions
        q = self.predictor(torch.cat([g1,g2])) #(2B, 256) in (view1, view2) order 

        # Get target predictions
        with torch.no_grad():
            self._update_target_network(mm)
            target_z = self.target_network(torch.cat([view2, view1])) #(2B, 256, 1, 1) (view2, view1 order)
            target_z = torch.flatten(target_z, 1) #(2B, 256)
        
        return q, target_z, atten_ab.view((b, h*w, h*w)), atten_ba.view((b, h*w, h*w))