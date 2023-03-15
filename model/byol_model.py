#-*- coding:utf-8 -*-
import torch
from .basic_modules import EncoderwithProjection, FCNMaskNetV2, Predictor, Masknet, SpatialAttentionMasknet,FCNMaskNet
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
from utils.kmeans.per_image import BatchwiseKMeans
from data.byol_transform import denormalize,rgb_to_hsv
from utils.mask_utils import inverse_align

class BYOLModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # online network
        self.online_network = EncoderwithProjection(config)

        # target network
        self.target_network = EncoderwithProjection(config)
        
        #mask net
        self.k_means_loss = config['loss']['type'] == 'k-means'
        self.masknet_on = config['model']['masknet']
        if self.masknet_on:
            self.masknet = FCNMaskNetV2()
        # predictor
        self.predictor = Predictor(config)
        self.over_lap_mask = config['data'].get('over_lap_mask',True)
        self._initializes_target_network()
        self._fpn = None # not actual FPN, but pesudoname to get c4, TODO: Change the confusing name
        self.slic_only = True
        self.slic_segments = config['data']['slic_segments']
        self.n_kmeans = config['data']['n_kmeans']
        #self.sub_sample_size = config['clustering']['sub_sample_size']
        self.per_image_k_means = config['clustering']['per_image']
        self.clustering_batchsize = config['clustering']['batch_size']
        if self.per_image_k_means:
            self.clustering_batchsize = 0 # ignore when per image=true
        if self.per_image_k_means or self.clustering_batchsize > 0:
            self.kmeans_class = BatchwiseKMeans
        else:
            self.kmeans_class = KMeans
        if self.n_kmeans < 9999:
            self.kmeans = self.kmeans_class(self.n_kmeans,)
        else:
            self.kmeans = None
        self.kmeans_gather = config['clustering']['gather'] # NOT TESTED
        self.no_slic  = config['clustering']['no_slic']
        self.rank = config['rank']
        self.add_views =  config['clustering']['add_views']
        self.use_pca = config['clustering']['use_pca']
        self.encoder_type = config['model']['backbone']['type']
        self.use_gt = config['data'].get('use_gt')
        #self.spatial_resolution = config['clustering']['spatial_resolution']
        if self.encoder_type == 'resnet50':
            self.feature_resolution = 7
        else:
            self.feature_resolution =  config['model']['backbone']['feature_resolution']
        # if self.add_views:
        #     assert self.no_slic, "add_views can be true only if explict slic is disabled"
        self.w_color = config['clustering']['weight_color']
        self.w_spatial = config['clustering']['weight_spatial']
        self.agg = AgglomerativeClustering(affinity='cosine',linkage='average',distance_threshold=0.2,n_clusters=None)
        self.agg_backup = AgglomerativeClustering(affinity='cosine',linkage='average',n_clusters=16)

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

    @torch.no_grad()
    def _update_mask_network(self, mm):
        """Momentum update of maks network"""
        #TODO: set this up properly, now masknet just use online encoder
        for param_q, param_k in zip(self.online_network.encoder.parameters(), self.masknet.encoder.parameters()):
            param_k.data.mul_(mm).add_(1. - mm, param_q.data)

    @property
    def fpn(self):
        if self._fpn:
            return self._fpn
        if self.encoder_type == 'vit' or self.encoder_type == 'vit-deconv':
            self._fpn = self.target_network.encoder
            return self._fpn
        elif self.encoder_type == 'resnet50':
            self._fpn = IntermediateLayerGetter(self.target_network.encoder, return_layers={'7':'out','6':'c4'})
            return self._fpn
        else:
            raise NotImplementedError
            
    def handle_flip(self,aligned_mask,flip):
        '''
        aligned_mask: B X C X 7 X 7
        flip: B 
        '''
        _,c,h,w = aligned_mask.shape
        b = len(flip)
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

    def get_spatial_mask(self,dimension,batch_size):
        coords = torch.stack(torch.meshgrid(torch.arange(dimension, device='cuda'), torch.arange(dimension, device='cuda'),indexing='ij'), 0)
        coords = coords[0] * dimension + coords[1]
        return coords.unsqueeze(0).repeat(batch_size,1,1) # H X W

    def get_feature(self,x):
        if self.encoder_type == 'vit-deconv':
            return self.fpn.forward_features(x)[-2] # 14 x 14 
        elif self.encoder_type == 'vit':
            r = self.fpn(x)
            return r # 14 x 14
        else:
            return self.fpn(x)['c4']
            
    def do_kmeans(self,raw_image,slic_mask,user_masknet,roi_t):
        if self.use_gt:
            converted_idx_b = to_binary_mask(slic_mask,self.n_kmeans,resize_to=(56,56))
            return converted_idx_b,slic_mask
        b = raw_image.shape[0]
        #feats = self.fpn(raw_image)['c4']
        feats = self.get_feature(raw_image) # B X C X H X W
        feats = F.normalize(feats,dim=1)
        
        if self.no_slic:
            raw_image_downsampled = F.adaptive_avg_pool2d(raw_image,self.spatial_resolution)
            raw_image_downsampled = denormalize(raw_image_downsampled)
            raw_image_downsampled = rgb_to_hsv(raw_image_downsampled)
            feats_downsample = F.adaptive_avg_pool2d(feats,self.spatial_resolution)
            coords = torch.stack(torch.meshgrid(torch.arange(self.spatial_resolution, device='cuda'), torch.arange(self.spatial_resolution, device='cuda'),indexing='ij'), 0)
            coords = coords[None].repeat(feats_downsample.shape[0], 1, 1, 1).float()
            super_pixel_pooled = torch.cat((raw_image_downsampled*self.w_color,feats_downsample,coords*self.w_spatial),dim=1).permute(0,2,3,1).flatten(1,2).contiguous() # B X 196 X 1027
        else:
            super_pixel = to_binary_mask(slic_mask,-1,resize_to=(14,14))
            if self.add_views:
                super_pixel = torch.cat([super_pixel,*self.roi_align(to_binary_mask(slic_mask,-1,resize_to=(56,56)),roi_t,14,56)])
            pooled, _ = maskpool(super_pixel,feats) #pooled B X 100 X d_emb
            super_pixel_pooled = pooled.detach()
        if not self.per_image_k_means:
            d_emb = super_pixel_pooled.shape[-1]
            super_pixel_pooled = super_pixel_pooled.view(-1,d_emb)
        if self.use_pca:
            _,_,v = torch.linalg.svd(super_pixel_pooled)
            super_pixel_pooled = super_pixel_pooled @ v[...,:self.use_pca]
        if self.kmeans_gather:
            # _,_,v = torch.linalg.sv(super_pixel_pooled)
            # super_pixel_pooled = super_pixel_pooled @ v[:,:256]
            super_pixel_pooled_large = gather_from_all(super_pixel_pooled)
        else:
            super_pixel_pooled_large = super_pixel_pooled
        if self.clustering_batchsize:
            assert b % self.clustering_batchsize == 0
            super_pixel_pooled_large = super_pixel_pooled_large.view(b//self.clustering_batchsize,-1,d_emb)
        labels = self.kmeans.fit_transform(super_pixel_pooled_large) # B X 100 
        if self.clustering_batchsize:
            labels = labels[0]
            labels = labels.reshape(b,-1)
        if self.per_image_k_means:
            labels = labels[0] #second return is centroid
        if self.kmeans_gather:
            start_idx =  self.rank *b
            end_idx = start_idx + b
            labels = labels[start_idx:end_idx]
        labels = labels.view(b,-1).to(slic_mask.device)
        if not self.no_slic:
            if self.add_views:
                raw_mask_target =  torch.einsum('bchw,bc->bchw',to_binary_mask(slic_mask,-1,(56,56)).repeat(3,1,1,1) ,labels).sum(1).long().detach()
            else:
                raw_mask_target =  torch.einsum('bchw,bc->bchw',to_binary_mask(slic_mask,-1,(56,56)) ,labels).sum(1).long().detach()
            if user_masknet:
                    # USE hirearchl clustering on outputs of masknet
                    raw_masks = self.masknet(feats)
                    converted_idx = self.get_label_map(raw_masks)
                    converted_idx = refine_mask(converted_idx,slic_mask,56).detach()
            else:
                    raw_masks = torch.ones(b,1,0,0).cuda() # to make logging happy
                    converted_idx = raw_mask_target
        else:
            #no slic
            converted_idx = labels.view(-1,self.spatial_resolution,self.spatial_resolution)
        converted_idx_b = to_binary_mask(converted_idx,self.n_kmeans,resize_to=(56,56))
        return converted_idx_b,converted_idx

    def roi_align(self,target,roi_t,out_size=7,mask_dim=56):
        device = target.device
        idx = torch.LongTensor([1,0,3,2]).to(device)
        rois_1 = [roi_t[j,:1,:4].index_select(-1, idx)*mask_dim for j in range(roi_t.shape[0])]
        rois_2 = [roi_t[j,1:2,:4].index_select(-1, idx)*mask_dim for j in range(roi_t.shape[0])]
        flip_1 = roi_t[:,0,4]
        flip_2 = roi_t[:,1,4]
        aligned_1 = self.handle_flip(ops.roi_align(target,rois_1,out_size),flip_1) # mask output is B X 16 X 7 X 7
        aligned_2 = self.handle_flip(ops.roi_align(target,rois_2,out_size),flip_2) # mask output is B X 16 X 7 X 7
        return aligned_1,aligned_2

    @staticmethod
    def subsample(seq, mask):
        if mask is None:
            return seq
        n, l = seq.shape[:2]
        _, l_mask = mask.shape
        x_arr = torch.arange(n).view(n, 1).repeat(1, l_mask)
        seq = seq[x_arr, mask]
        return seq

    def sample_masks(self,maska,maskb,n_masks=16):
        batch_size=maska.shape[0]
        mask_exists = torch.greater(maska.sum(-1), 1e-3) &  torch.greater(maskb.sum(-1), 1e-3) # N X L
        sel_masks = mask_exists.float() + 0.00000000001
        sel_masks = sel_masks / sel_masks.sum(1, keepdims=True)
        sel_masks = torch.log(sel_masks)
        
        dist = torch.distributions.categorical.Categorical(logits=sel_masks)
        mask_ids = dist.sample([n_masks]).T
        #breakpoint()
        return self.subsample(maska,mask_ids), self.subsample(maskb,mask_ids)
    
    def region_pool(self,z_dict,key='c5',dims=dict(c6=4,c7=2,c8=1)):
        feature1,feature2,msk = z_dict[key]
        for k,v in dims.items():
            z_dict[k] = (
                F.adaptive_avg_pool2d(feature1,(v,v)),
                F.adaptive_avg_pool2d(feature2,(v,v)),
                (F.adaptive_avg_pool2d(msk,(v,v)) >0).float(),
            )
        return z_dict
    
    def forward(self, view1, view2, mm, input_masks,raw_image,roi_t,slic_mask,user_masknet=False,full_view_prior_mask=None,clustering_k=64):
        im_size = view1.shape[-1]
        b = view1.shape[0] # batch size
        # assert im_size == 224
        #print(view1.shape)
        online_z = self.online_network(torch.cat([view1, view2], dim=0))

        #q,pinds = self.predictor()
        # target network forward
        rot_t_twoview = torch.cat([roi_t[:,[0,1,2]],roi_t[:,[1,0,2]]])
        with torch.no_grad():
            self._update_target_network(mm)
            target_z = self.target_network(torch.cat([view2, view1], dim=0))
            target_z = {k:v.detach().clone() for k,v in target_z.items()}
        z_dict = {}
        for k,v in online_z.items():
            z_dict[k] = inverse_align(v,target_z[k],rot_t_twoview,out_dim=v.shape[-1])
        z_dict = self.region_pool(z_dict)
        for k,v in z_dict.items():
            f1,f2,msk = v
            z_dict[k] = (self.predictor(f1),f2,msk)
        return z_dict

