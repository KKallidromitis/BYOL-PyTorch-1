#-*- coding:utf-8 -*-
from operator import imod
from jax import mask
import torch
from .basic_modules import EncoderwithProjection, FCNMaskNetV2, Predictor, Masknet, SpatialAttentionMasknet,FCNMaskNet
from utils.mask_utils import convert_binary_mask,sample_masks,to_binary_mask,maskpool
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

class BYOLModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # online network
        self.online_network = EncoderwithProjection(config)

        # target network
        self.target_network = EncoderwithProjection(config)
        
        #mask net
        
        self.masknet_on = config['model']['masknet']
        if self.masknet_on:
            self.masknet = FCNMaskNetV2()
        # predictor
        self.predictor = Predictor(config)
        self.over_lap_mask = config['data'].get('over_lap_mask',True)

        self._initializes_target_network()
        self._fpn = None
        self.slic_only = True
        self.n_kmeans = 16
        self.kmeans = KMeans(self.n_kmeans,)
        self.kmeans_gather = False
        self.rank = config['rank']
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
        else:
            self._fpn = IntermediateLayerGetter(self.target_network.encoder, return_layers={'7':'out','6':'c4'})
            return self._fpn
            
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
        
        

    def forward(self, view1, view2, mm, input_masks,raw_image,roi_t,slic_mask,user_masknet=False):
        # online network forward
        #import ipdb;ipdb.set_trace()
        #breakpoint()
        im_size = view1.shape[-1]
        assert im_size == 224
        idx = torch.LongTensor([1,0,3,2]).cuda()
         # B X 2048 X 7 X 7
        ## Use masknet label map
        #breakpoint()
        if self.masknet_on:
            if self.slic_only:
                feats = self.fpn(raw_image)['c4']
                masks = self.masknet(feats.detach())
                #breakpoint()
                b,slic_h,slic_w = slic_mask.shape
                slic_mask = slic_mask.view(b,slic_h,slic_w)  # B X H_mask X W_mask
                super_pixel = to_binary_mask(slic_mask,100,resize_to=(14,14))
                pooled, _ = maskpool(super_pixel,feats)
                # do kemans
                super_pixel_pooled = pooled.view(-1,1024).detach()
                if self.kmeans_gather:
                    super_pixel_pooled_large = gather_from_all(super_pixel_pooled)
                else:
                    super_pixel_pooled_large = super_pixel_pooled
                similarity_based = False
                #dataset = KMeansDataset(super_pixel_pooled_large, similarity_based=similarity_based)
                #dataloader = DataLoader(dataset, batch_size=1024, num_workers=0, shuffle=True)
                labels = self.kmeans.fit_transform(F.normalize(super_pixel_pooled_large,dim=-1))
                #breakpoint()
                if self.kmeans_gather:
                    start_idx =  self.rank *b
                    end_idx = start_idx + b
                    labels = labels[start_idx:end_idx]
                    #labels = self.kmeans.transform_tensor(super_pixel_pooled_large)
                #OLD NUMPY KMEANS IMPLEM
                # super_pixel_pooled = pooled.detach().cpu().numpy().reshape(-1,1024)
                # kmeans_superpixel = MiniBatchKMeans(n_clusters=32, random_state=0,batch_size=1024).fit(super_pixel_pooled)
                #label_map = kmeans_superpixel.labels_.reshape(b,-1)
                #label_map =torch.LongTensor(label_map).cuda()
                labels = labels.view(b,-1)
                #breakpoint()
                #breakpoint()
                #converted_idx_b = to_binary_mask(converted_idx,self.n_kmeans)
                raw_masks = F.normalize(masks,dim=1)
                raw_mask_target =  torch.einsum('bchw,bc->bchw',to_binary_mask(slic_mask,100,(56,56)) ,labels).sum(1).long().detach()
                if user_masknet:
                    converted_idx = self.get_label_map(raw_masks)
                else:
                    converted_idx = raw_mask_target
                converted_idx_b = to_binary_mask(converted_idx,16)
                #breakpoint()
                mask_dim = 56
            else:
                raw_encoding = self.target_network.encoder(raw_image)
                b,slic_h,slic_w = slic_mask.shape
                masks = self.masknet(raw_encoding.detach()) # raw_image: B X 3 X H X W, ->Mask B X 16 X H_mask X W_mask
                _,_,_,mask_dim = masks.shape
                slic_mask = slic_mask.reshape(b,slic_h,slic_w)  # B X H_mask X W_mask
                slic_mask = to_binary_mask(slic_mask,100,(mask_dim,mask_dim))  # B X 100 X H_mask X W_mask
                masknet_label_map = torch.argmax(masks,1).detach()
                masknet_label_map = to_binary_mask(masknet_label_map,16) # binary B X 16 X H_mask X W_mask
                pooled,_ =maskpool(slic_mask,masknet_label_map) # B X NUM_SLIC X N_MASKS
                pooled_ids = torch.argmax(pooled,-1) # B X NUM_SLIC  X 1 => label map
                converted_idx = torch.einsum('bchw,bc->bchw',slic_mask ,pooled_ids).sum(1).long().detach() #label map in hw space
                raw_masks = masks
                raw_mask_target = converted_idx
                converted_idx_b = to_binary_mask(converted_idx,16)
            #breakpoint()
            rois_1 = [roi_t[j,:1,:4].index_select(-1, idx)*mask_dim for j in range(roi_t.shape[0])]
            rois_2 = [roi_t[j,1:2,:4].index_select(-1, idx)*mask_dim for j in range(roi_t.shape[0])]
            flip_1 = roi_t[:,0,4]
            flip_2 = roi_t[:,1,4]
            #detached_mask = masks.detach()
            aligned_1 = self.handle_flip(ops.roi_align(converted_idx_b,rois_1,7),flip_1) # mask output is B X 16 X 7 X 7
            aligned_2 = self.handle_flip(ops.roi_align(converted_idx_b,rois_2,7),flip_2) # mask output is B X 16 X 7 X 7
            #breakpoint()
            #breakpoint()
            mask_b,mask_c,h,w =aligned_1.shape
            aligned_1 = aligned_1.reshape(mask_b,mask_c,h*w)
            aligned_2 = aligned_2.reshape(mask_b,mask_c,h*w)
            # aligned_1 = F.softmax(aligned_1,dim=-2)
            # aligned_2 = F.softmax(aligned_2,dim=-2)
            #breakpoint()
            if self.over_lap_mask:
                intersection = input_masks.float()
                intersec_masks_1 = F.adaptive_avg_pool2d(intersection[:,0,...],(h,w)).repeat(1,mask_c,1,1).reshape(mask_b,mask_c,h*w)
                intersec_masks_2 = F.adaptive_avg_pool2d(intersection[:,1,...],(h,w)).repeat(1,mask_c,1,1).reshape(mask_b,mask_c,h*w)
                aligned_1 = aligned_1 * intersec_masks_1
                aligned_2 = aligned_2 * intersec_masks_2
            mask_ids = None
            masks = torch.cat([aligned_1, aligned_2])
            masks_inv = torch.cat([aligned_2, aligned_1])
            num_segs = torch.FloatTensor([x.unique().shape[0] for x in converted_idx]).mean()
        else:
            #breakpoint()
            b,_,mh,mw = input_masks.shape
            input_masks = input_masks.permute(1,0,2,3).reshape(2*b,1,mh,mw)
            masks = convert_binary_mask(input_masks)
            masks,mask_ids = sample_masks(masks)
            mask_batch_size = masks.shape[0] // 2
            masks_a = masks[:mask_batch_size]
            masks_b = masks[mask_batch_size:]
            masks_inv = torch.cat([masks_b, masks_a])
            num_segs = torch.FloatTensor([x.unique().shape[0] for x in mask_ids]).mean()
            raw_masks = None
            raw_mask_target = None #B X H X W (input mask)
            converted_idx = None


        q,pinds = self.predictor(*self.online_network(torch.cat([view1, view2], dim=0),masks.to('cuda'),mask_ids,mask_ids))
        # target network forward
        with torch.no_grad():
            self._update_target_network(mm)
            target_z, tinds = self.target_network(torch.cat([view2, view1], dim=0),masks_inv.to('cuda'),mask_ids,mask_ids)
            target_z = target_z.detach().clone()
        

        return q, target_z, pinds, tinds,masks,raw_masks,raw_mask_target,num_segs,converted_idx

    def _legacy_forward(self, view1, view2, mm, masks):
        # online network forward
        #import ipdb;ipdb.set_trace()
        #breakpoint()
        masks = torch.cat([ masks[:,i,:,:,:] for i in range(masks.shape[1])])
        masks = convert_binary_mask(masks)
        masks,mask_ids = sample_masks(masks)
        mnet = self.masknet
        q,pinds = self.predictor(*self.online_network(torch.cat([view1, view2], dim=0),masks.to('cuda'),mask_ids,mnet))
        mask_batch_size = masks.shape[0] // 2
        masks_a = masks[:mask_batch_size]
        masks_b = masks[mask_batch_size:]
        # target network forward
        with torch.no_grad():
            self._update_target_network(mm)
            target_z, tinds = self.target_network(torch.cat([view2, view1], dim=0),torch.cat([masks_b,masks_a]).to('cuda'),mask_ids,mnet)
            target_z = target_z.detach().clone()

        return q, target_z, pinds, tinds,masks
