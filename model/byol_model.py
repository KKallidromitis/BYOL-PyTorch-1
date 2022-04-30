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

    def forward(self, view1, view2, mm):

        def cosine_attention(dense_emb,global_emb):
            '''
            Computes cosine similarity between dense_emb and global_emb
            Args:
                dense_emb: (B, H, W, C)
                global_emb: (B, C)
            Returns
                atten: (B, H, W)
            '''
            dense_emb = F.normalize(dense_emb,dim=-1)
            global_emb = F.normalize(global_emb,dim=-1)
            atten = torch.einsum('bhwc,bc->bhw',dense_emb, global_emb)
            atten = F.relu(atten, inplace=True)
            return atten

        online_encoder = self.online_network.encoder
        online_projector = self.online_network.projetion

        # Get Online Encoding
        b = view1.size(0)
        online_encoding = online_encoder(torch.cat([view1, view2])) #(2B, 2048, 7, 7)
        h, w = online_encoding.shape[2:] #(7, 7)
        global_pooled = F.adaptive_avg_pool2d((online_encoding), (1,1)) #(2B, 2048, 1, 1)
        online_encoding = torch.permute(online_encoding, (0,2,3,1)) #(2B, 7,7, 2048)
        online_encoding = torch.flatten(online_encoding, 0, 2) #(2B*49, 2048)
        global_pooled = torch.flatten(global_pooled, 1) #(2B, 2048)

        dense_projection = online_projector(online_encoding) #(2B*49, 256)
        global_projection = online_projector(global_pooled) # (2B, 256)
        dense_v1, dense_v2 = dense_projection[:b*h*w], dense_projection[b*h*w:]  #(B*49, 256), (B*49, 256)
        dense_v1, dense_v2 = torch.reshape(dense_v1, (b, h, w, -1)), torch.reshape(dense_v2, (b, h, w, -1)) #(B, 7, 7, 256), (B, 7, 7, 256)
        global_v1, global_v2 = global_projection[:b], global_projection[b:] #(B, 256), (B, 256)

        atten_ab = cosine_attention(dense_v1, global_v2) #(B, H, W)
        atten_ba = cosine_attention(dense_v2, global_v1) #(B, H, W)

        q1 = torch.einsum("bhwc, bhw -> bc", dense_v1, atten_ab) #Use "attention weighted" dense projection (B, 256)
        q2 = torch.einsum("bhwc,bhw -> bc", dense_v2, atten_ba) #(B, 256)
        q = self.predictor(torch.cat([q1, q2])) #(2B, 256)

        # Get target encodings
        with torch.no_grad():
            self._update_target_network(mm)
            target_z = self.target_network(torch.cat([view2, view1], dim=0)).detach().clone() #(2B, 256)
        
        return q, target_z
