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
        # self._fpn = None # not actual FPN, but pesudoname to get c4, TODO: Change the confusing name
        # self.n_kmeans = config['data']['n_kmeans']
        # self.encoder_type = config['model']['backbone']['type']
        self.feature_resolution =  config['model']['backbone']['feature_resolution']
        # if self.n_kmeans < 9999:
        #     self.kmeans = KMeans(self.n_kmeans,)
        # else:
        #     self.kmeans = None
        # self.kmeans_gather = False # NOT TESTED
        # self.rank = config['rank']
        self.agg = AgglomerativeClustering(affinity='cosine',linkage='average',distance_threshold=0.2,n_clusters=None)
        self.agg_backup = AgglomerativeClustering(affinity='cosine',linkage='average',n_clusters=16)

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

    # @torch.no_grad()
    # def _update_mask_network(self, mm):
    #     """Momentum update of maks network"""
    #     #TODO: set this up properly, now masknet just use online encoder
    #     for param_q, param_k in zip(self.online_network.encoder.parameters(), self.masknet.encoder.parameters()):
    #         param_k.data.mul_(mm).add_(1. - mm, param_q.data)

    # @property
    # def fpn(self):
    #     if self._fpn:
    #         return self._fpn
    #     else:
    #         if self.encoder_type == 'vit':
    #             self._fpn = self.target_encoder
    #         else:
    #             self._fpn = IntermediateLayerGetter(self.target_encoder, return_layers={'7':'out','6':'c4'})
    #         return self._fpn

    # def get_feature(self,x):
    #     if self.encoder_type == 'vit':
    #         return self.fpn(x)
    #     else:
    #         return self.fpn(x)['c4']
             
    def handle_flip(self,aligned_mask,flip):
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

    def get_label_map(self,masks):
        #
        b,c,h,w = masks.shape
        batch_data = masks.permute(0,2,3,1).reshape(b,h*w,32).detach().cpu()
        labels = []
        for data in batch_data:
            agg = self.agg.fit(data)
            if np.max(agg.labels_)>15:
                agg = self.agg_backup.fit(data)
            label = agg.labels_.reshape(h,w)
            labels.append(label)
        labels = np.stack(labels)
        labels = torch.LongTensor(labels).cuda()
        return labels

    # def do_kmeans(self,raw_image,slic_mask,user_masknet):
    #     b = raw_image.shape[0]
    #     #feats = self.fpn(raw_image)['c4']
    #     #feats = self.online_network.encoder.forward_features(raw_image) # N X 196 X 768
    #     feats = self.get_feature(raw_image) # N X C X 14 X 14
    #     _,d_feature,_,_ = feats.shape
    #     super_pixel = to_binary_mask(slic_mask,-1,resize_to=(14,14))
    #     pooled, _ = maskpool(super_pixel,feats) #pooled B X 100 X d_emb
    #     super_pixel_pooled = pooled.view(-1,d_feature).detach()
    #     if self.kmeans_gather:
    #         super_pixel_pooled_large = gather_from_all(super_pixel_pooled)
    #     else:
    #         super_pixel_pooled_large = super_pixel_pooled
    #     #breakpoint()
    #     labels = self.kmeans.fit_transform(F.normalize(super_pixel_pooled_large,dim=-1)) # B X 100
    #     if self.kmeans_gather:
    #         start_idx =  self.rank *b
    #         end_idx = start_idx + b
    #         labels = labels[start_idx:end_idx]
    #     labels = labels.view(b,-1)
    #     raw_mask_target =  torch.einsum('bchw,bc->bchw',to_binary_mask(slic_mask,-1,(56,56)) ,labels).sum(1).long().detach()
    #     if user_masknet:
    #             # USE hirearchl clustering on outputs of masknet
    #             raw_masks = self.masknet(feats)
    #             converted_idx = self.get_label_map(raw_masks)
    #             converted_idx = refine_mask(converted_idx,slic_mask,56).detach()
    #     else:
    #             raw_masks = torch.ones(b,1,0,0).cuda() # to make logging happy
    #             converted_idx = raw_mask_target
    #     converted_idx_b = to_binary_mask(converted_idx,self.n_kmeans)
    #     return converted_idx_b,converted_idx
     
    def forward(self, view1, view2, mm, input_masks,raw_image,roi_t,slic_mask,user_masknet=False,full_view_prior_mask=None,clustering_k=64):
        im_size = view1.shape[-1]
        b = view1.shape[0] # batch size
        assert im_size == 224
        idx = torch.LongTensor([1,0,3,2]).cuda()
        # reset k means if necessary
        # if self.n_kmeans != clustering_k:
        #     self.n_kmeans = clustering_k
        #     if self.n_kmeans < 9999:
        #         self.kmeans = KMeans(self.n_kmeans,)
        #     else:
        #         self.kmeans = None
        # Get spanning view embeddings
        # with torch.no_grad():
            # if self.n_kmeans < 9999:
            #     converted_idx_b,converted_idx = self.do_kmeans(raw_image,slic_mask,user_masknet) # B X C X 56 X 56, B X 56 X 56
            # else:
            #     converted_idx_b = to_binary_mask(slic_mask,-1,(56,56))
            #     converted_idx = torch.argmax(converted_idx_b,1)
            # raw_masks = torch.ones(b,1,0,0).cuda()
            # raw_mask_target = converted_idx
            # #TODO: Move this to config, mask size before downsample
            # mask_dim = 56
            # #TODO: Move this to a separate method just as in selective search branch
            # rois_1 = [roi_t[j,:1,:4].index_select(-1, idx)*mask_dim for j in range(roi_t.shape[0])]
            # rois_2 = [roi_t[j,1:2,:4].index_select(-1, idx)*mask_dim for j in range(roi_t.shape[0])]
            # flip_1 = roi_t[:,0,4]
            # flip_2 = roi_t[:,1,4]
            # aligned_1 = self.handle_flip(ops.roi_align(converted_idx_b, rois_1, self.feature_resolution), flip_1) # mask output is B X 16 X 7 X 7
            # aligned_2 = self.handle_flip(ops.roi_align(converted_idx_b, rois_2, self.feature_resolution), flip_2) # mask output is B X 16 X 7 X 7

        aligned = self.decoder(previous_z)
        mask_b, mask_c, mask_s = aligned.shape
        sum_aligned = aligned.sum(dim = 1, keepdim = True)
        # aligned[sum_aligned == 0] = 1.0
        # sum_aligned[sum_aligned == 0] = mask_c
        aligned = aligned / sum_aligned # how to manage division by zero
        aligned_1 = aligned[mask_b//2: , :, :]
        aligned_2 = aligned[0:mask_b//2, :, :]

        # If this is on, only mask in intersetved area will be calculated
        # if self.over_lap_mask:
        #     intersection = input_masks.float()
        #     intersec_masks_1 = F.adaptive_avg_pool2d(intersection[:,0,...],(h,w)).repeat(1,mask_c,1,1).reshape(mask_b,mask_c,h*w)
        #     intersec_masks_2 = F.adaptive_avg_pool2d(intersection[:,1,...],(h,w)).repeat(1,mask_c,1,1).reshape(mask_b,mask_c,h*w)
        #     aligned_1 = aligned_1 * intersec_masks_1
        #     aligned_2 = aligned_2 * intersec_masks_2

        num_segs = torch.FloatTensor([x.unique().shape[0] for x in converted_idx]).mean()

        mask_ids = None
        masks = torch.cat([aligned_1, aligned_2])
        masks_inv = torch.cat([aligned_2, aligned_1])
        online_enc_q = self.online_encoder(torch.cat([view1, view2])) # Should the list be unpacked by '*'?
        q, pinds = self.projector(online_enc_q, masks.to('cuda'), mask_ids)
        q = self.predictor(q)

        # target network forward
        with torch.no_grad():
            self._update_target_network(mm)
            target_enc_z = self.target_encoder(torch.cat([view2, view1]))
            target_z, tinds = self.target_projector(target_enc_z, masks_inv.to('cuda'), mask_ids)
            target_z = target_z.detach().clone()
        

        return q, target_z, pinds, tinds, masks, raw_masks, raw_mask_target, num_segs, converted_idx