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
        self.feature_resolution =  config['model']['backbone']['feature_resolution']
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

    def handle_flip(self, aligned_mask, flip):
        '''
        aligned_mask: B X C X 7 X 7
        flip: B 
        '''
        _,c,h,w = aligned_mask.shape
        b = len(flip)
        #breakpoint()
        flip = flip.repeat(c*h*w).reshape(c,h,w,b) # C X H X W X B
        flip = flip.permute(3,0,1,2)
        flipped = aligned_mask.flip(-1)
        out = torch.where(flip==1,flipped,aligned_mask)
        return out

    def generate_init_mask(self, batch_size, radius_rate = 0.5):
        mask_size = config['model']['decoder']['output_dim']
        h = w = np.sqrt(mask_size)
        r = np.amax(np.array([h/2*radius_rate, 0.5]))
        k = np.arange(mask_size)
        i, j = k//w, k%w
        d = (i - (h - 1)/2)**2 + (j - (w - 1)/2)**2
        mask_c1 = np.exp(-d*np.log(2)/r**2)
        mask_c2 = 1.0 - mask_c1
        mask = np.vstack([mask_c2, mask_c1])
        mask = np.repeat(mask[np.newaxis, :, :], batch_size, axis = 0)
        return torch.from_numpy(mask)

    def form_mask(self, mask_raw, roi_t)
        # regularize the sum of the masks to 1 in the cluster direction
        mask_b, mask_c, mask_s = mask_raw.shape
        mask_raw += 1.0*10**(-6) # epsilon to prevent division by zero
        sum_mask_raw = mask_raw.sum(dim = 1, keepdim = True)
        mask_raw = mask_raw / sum_mask_raw

        #TODO: Move this to config, mask size before downsample
        mask_dim = 28
        rois_1 = [roi_t[j,:1,:4].index_select(-1, idx)*mask_dim for j in range(roi_t.shape[0])]  # roi_t = B X 3 X 5, rois_1 = B X 1 X 4
        rois_2 = [roi_t[j,1:2,:4].index_select(-1, idx)*mask_dim for j in range(roi_t.shape[0])]
        flip_1 = roi_t[:,0,4]
        flip_2 = roi_t[:,1,4]
        aligned_1 = self.handle_flip(ops.roi_align(mask_raw, rois_1, self.feature_resolution), flip_1) # mask output is B X 16 X 7 X 7
        aligned_2 = self.handle_flip(ops.roi_align(mask_raw, rois_2, self.feature_resolution), flip_2) # mask output is B X 16 X 7 X 7

        return aligned_1, aligned_2, mask_raw

    def forward(self, view1, view2, mm, roi_t, pre_enc_q=None, pre_target_enc_z=None, pre_target_z=None):
        im_size = view1.shape[-1]
        batch_size = view1.shape[0]
        assert im_size == 224

        if pre_target_z is None:
            mask_raw = self.generate_init_mask(batch_size)
        else:
            mask_raw = self.decoder(pre_target_z)
        mask_1, mask_2, mask_raw = self.form_mask(mask_raw, roi_t)
        masks = torch.cat([mask_1, mask_2])
        masks_inv = torch.cat([mask_2, mask_1])

        # online network forward
        if pre_enc_q is None:
            enc_q = self.online_encoder(torch.cat([view1, view2]))
        else:
            enc_q = pre_enc_q.detach()
        q = self.projector(enc_q, masks.to('cuda'))
        q = self.predictor(q)

        # target network forward
        with torch.no_grad():
            self._update_target_network(mm)
            if pre_target_enc_z is None:
                target_enc_z = self.target_encoder(torch.cat([view2, view1]))
            else:
                target_enc_z = pre_target_enc_z.detach()
            target_z = self.target_projector(target_enc_z, masks_inv.to('cuda'))
            # target_z = target_z.detach().clone() ##nishio## Is this necessary? 

        return q, enc_q, target_z, target_enc_z, masks, mask_raw #, mask_1, mask_2
        # return q, target_z, pinds, tinds,masks,raw_masks,raw_mask_target,num_segs,converted_idx