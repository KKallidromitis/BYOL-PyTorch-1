#-*- coding:utf-8 -*-
import torch
from .basic_modules import EncoderwithProjection, Predictor,cosine_attention
import torch.nn.functional as F

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
        predictor = self.predictor.predictor
        encoder_a = self.online_network.encoder
        projector_a = self.online_network.projetion
        encoder_b = self.online_network.encoder
        projector_b = self.online_network.projetion
        emb_a = encoder_a(torch.cat([view1, view2])) # 2B X 2048 X 7 X 7
        b,c,h,w = emb_a.shape # embedding shape
        hb = b//2
        emb_aa,emb_ab = emb_a[:hb],emb_a[hb:] # B X 2048 X 7 X 7
        emb_b = encoder_b(torch.cat([view2, view1])) # 2B X 2048 X 7 X 7
        emb_a_pooling = emb_a.mean(dim=(-1,-2)) # 2B X 2048
        emb_aa_pooling,emb_ab_pooling = emb_a_pooling[:hb],emb_a_pooling[hb:] # B X 2048
        atten_aa = cosine_attention(emb_aa,emb_ab_pooling) # B X 7 X 7
        atten_ab = cosine_attention(emb_ab,emb_aa_pooling) # B X 7 X 7
        # Do normalize (using linear as an example)
        atten_a = torch.cat([atten_aa,atten_ab])
        atten_a = (atten_a + 1.) / 2. # 2B X 7 X 7 in (0,1)
        atten_a = F.normalize(atten_a,dim=(-1,-2),p=1) # normalize by "mask area", effectivly /= mask_area
        atten_b = torch.cat([atten_a[hb:],atten_a[:hb]]) # invert order
        emb_a = torch.einsum('bcxy,bxy->bc',emb_a,atten_a)#.view(b,1,c)  # 2B X 1 X 2048
        emb_b = torch.einsum('bcxy,bxy->bc',emb_b,atten_b)#.view(b,1,c)  # 2B X 1 X 2048 
        proj_a = projector_a(emb_a)  # 2B X 256 
        proj_b = projector_b(emb_b) # 2B X 256 
        prediction_a = predictor(proj_a)
        # assign names to be consitent for readibility
        q = prediction_a # B X 256
        target_z = proj_b # B X 256
        #Ref return q, target_z, pinds, tinds,masks,raw_masks,raw_mask_target,num_segs,converted_idx
        return q,target_z,atten_a,atten_b
