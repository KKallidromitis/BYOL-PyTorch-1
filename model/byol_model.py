#-*- coding:utf-8 -*-
import torch
from .basic_modules import Encoder, Projector, Decoder, FCNMaskNetV2, Predictor, Masknet, SpatialAttentionMasknet,FCNMaskNet
from utils.mask_utils import convert_binary_mask,sample_masks,to_binary_mask,maskpool,refine_mask
from torchvision import ops
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from sklearn.cluster import AgglomerativeClustering
from utils.kmeans.minibatchkmeans import MiniBatchKMeans
from utils.kmeans.kmeans import KMeans
from utils.kmeans.dataset import KMeansDataset
from utils.distributed_utils import gather_from_all
from torch.utils.data import DataLoader
import numpy as np

class PredR2O(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # online network
        self.online_encoder = Encoder(config)
        self.online_projector = Projector(config)
        # target network
        self.target_encoder = Encoder(config)
        self.target_projector = Projector(config)
        # decoder
        self.decoder = Decoder(config)
        
        ## mask net
        # self.masknet_on = config['model']['masknet']
        # if self.masknet_on:
        #     self.masknet = FCNMaskNetV2()

        # predictor
        self.predictor = Predictor(config)
        # self.over_lap_mask = config['data'].get('over_lap_mask',True)
        self._initializes_target_network()
        # self.feature_resolution =  config['model']['backbone']['feature_resolution']
        # self.rank = config['rank']
        # self.agg = AgglomerativeClustering(affinity='cosine',linkage='average',distance_threshold=0.2,n_clusters=None)
        # self.agg_backup = AgglomerativeClustering(affinity='cosine',linkage='average',n_clusters=16)

    @torch.no_grad()
    def _initializes_target_network(self):
        networks = [[self.online_encoder, self.target_encoder], [self.online_projector, self.target_projector]]
        for online_network, target_network in networks:
            for param_q, param_k in zip(online_network.parmeters(), target_network.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False     # not update by gradient

    @torch.no_grad()
    def _update_target_network(self, mm):
        """Momentum update of target network"""
        networks = [[self.online_encoder, self.target_encoder], [self.online_projector, self.target_projector]]
        for online_network, target_network in networks:
            for param_q, param_k in zip(online_network.parameters(), target_network.parameters()):
                param_k.data.mul_(mm).add_(1. - mm, param_q.data)

    def generate_init_mask(self, batch_size, radius_rate = 0.5):
        mask_size = config['model']['decoder']['output_dim']
        h = w = np.sqrt(mask_size)
        r = h/2*radius_rate
        k = np.arange(mask_size)
        i, j = k//w, k%w
        d = (i - (h - 1)/2)**2 + (j - (w - 1)/2)**2
        mask_c1 = np.exp(-d*np.log(2)/r**2)
        mask_c2 = 1 - mask_c1
        mask = np.vstack([mask_c2, mask_c1])
        mask = np.repeat(mask[np.newaxis, :, :], batch_size*2, axis = 0)
        return torch.from_numpy(mask)

    def forward(self, view1, view2, mm, pre_target_z=None):
        im_size = view1.shape[-1]
        batch_size = view1.shape[0]
        assert im_size == 224

        if pre_target_z is None:
            mask_raw = self.generate_init_mask(batch_size)
        else:
            mask_raw = self.decoder(pre_target_z) ##nishio## is it neccesary to regularize the output by sigmoid function?
        mask_b, mask_c, mask_s = mask_raw.shape
        sum_mask_raw = mask_raw.sum(dim = 1, keepdim = True)
        # mask[sum_mask == 0] = 1.0
        # sum_mask[sum_mask == 0] = mask_c
        mask_raw = mask_raw / sum_mask_raw ##nishio## how to handle division by zero
        ##nisho## how to crop the mask_raw to generate mask_1, 2
        mask_1 = mask_raw[mask_b//2: , :, :]
        mask_2 = mask_raw[0:mask_b//2, :, :]

        mask_ids = None
        masks = torch.cat([mask_1, mask_2])
        masks_inv = torch.cat([mask_2, mask_1])
        online_enc_q = self.online_encoder(torch.cat([view1, view2])) ##nishio## Should the list be unpacked by '*'?
        q, pinds = self.projector(online_enc_q, masks.to('cuda'), mask_ids)
        q = self.predictor(q)

        # target network forward
        with torch.no_grad():
            self._update_target_network(mm)
            target_enc_z = self.target_encoder(torch.cat([view2, view1]))
            target_z, tinds = self.target_projector(target_enc_z, masks_inv.to('cuda'), mask_ids)
            target_z = target_z.detach().clone()
        

        return q, target_z, pinds, tinds, masks