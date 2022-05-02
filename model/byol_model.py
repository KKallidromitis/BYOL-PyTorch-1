#-*- coding:utf-8 -*-
from operator import imod
from jax import mask
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

class BYOLModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # online network
        self.online_network = EncoderwithProjection(config)

        # target network
        self.target_network = EncoderwithProjection(config)
        
        self.predictor = Predictor(config)
        self.over_lap_mask = config['data'].get('over_lap_mask',True)
        self._initializes_target_network()
        self.slic_only = True
        self.n_kmeans = 64
        self.n_mask = 64
        self.rank = config.get('rank',0)

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
        
    def forward_attention(self, view1, view2, mm, input_masks,raw_image,roi_t,slic_mask,user_masknet=False,full_view_prior_mask=None):
        def cosine_attention(pixels,ref_vec):
            '''
            ref_vec: B X dim_emb
            pixels: B X H X W X dim_emb
            '''
            ref_vec = F.normalize(ref_vec,dim=-1)
            pixels = F.normalize(pixels,dim=-1)
            atten = torch.einsum('bcxy,bc->bxy',pixels,ref_vec)
            return atten
        predictor = self.predictor.predictor
        encoder_a = self.online_network.encoder
        projector_a = self.online_network.projetion
        encoder_b = self.online_network.encoder
        projector_b = self.online_network.projetion
        emb_a = encoder_a(torch.cat([view1, view2])) # 2B X 2048 X 7 X 7
        b,c,h,w = emb_a.shape # embedding shape
        emb_b = encoder_b(torch.cat([view2, view1])) # 2B X 2048 X 7 X 7
        emb_a_pooling = emb_a.mean(dim=(-1,-2)) # 2B X 2048
        emb_b_pooling = emb_b.mean(dim=(-1,-2)) # 2B X 2048
        breakpoint()
        atten_a = cosine_attention(emb_a,emb_b_pooling) # 2B X 7 X 7
        atten_b = cosine_attention(emb_b,emb_a_pooling) # 2B X 7 X 7
        # Do normalize (using linear as an example)
        atten_a = (atten_a + 1.) / 2. # 2B X 7 X 7 in (0,1)
        atten_b = (atten_b + 1.) / 2. # 2B X 7 X 7 in (0,1)
        atten_a = F.normalize(atten_a,dim=(-1,-2),p=1) # normalize by "mask area", effectivly /= mask_area
        atten_b = F.normalize(atten_a,dim=(-1,-2),p=1) 
        emb_a = torch.einsum('bcxy,bxy->bc',emb_a,atten_a).view(b,1,c)  # 2B X 1 X 2048
        emb_b = torch.einsum('bcxy,bxy->bc',emb_b,atten_b).view(b,1,c)  # 2B X 1 X 2048 
        proj_a = projector_a(emb_a)  # 2B X 256 
        proj_b = projector_b(emb_b) # 2B X 256 
        prediction_a = predictor(proj_a)
        # assign names to be consitent for readibility
        q = prediction_a # B X 256
        target_z = proj_b # B X 256
        #Ref return q, target_z, pinds, tinds,masks,raw_masks,raw_mask_target,num_segs,converted_idx
        return q,target_z,atten_a,atten_b
        

    def kmeans_slic(self,feats,slic_mask,gather=False):
        b,slic_h,slic_w = slic_mask.shape
        super_pixel = to_binary_mask(slic_mask,100,resize_to=(14,14))
        pooled, _ = maskpool(super_pixel,feats)
        # do kemans
        super_pixel_pooled = pooled.view(-1,1024).detach()
        if gather:
            super_pixel_pooled_large = gather_from_all(super_pixel_pooled)
        else:
            super_pixel_pooled_large = super_pixel_pooled
        labels = self.kmeans.fit_transform(F.normalize(super_pixel_pooled_large,dim=-1))
        if gather:
            start_idx =  self.rank *b
            end_idx = start_idx + b
            labels = labels[start_idx:end_idx]
        labels = labels.view(b,-1)
        return None
    
    def binarize_full_view_masks(self,masks,target_dim=56):
        max_idx = (torch.max(masks)+1).item()
        b,c,h,w = masks.shape # c-> Num of label maps, [slic,selective search, etc ]
        #breakpoint()
        #breakpoint()
        masks = to_binary_mask(masks.reshape(b*c,h,w),max_idx,(target_dim,target_dim)).reshape(b,c*max_idx,target_dim,target_dim)
        return masks

    def roi_transforms(self,roi_t,raw_view_masks):
        '''
        rot_t: record of transfroms
        mask_dim: dimension of converted_idx_b 
        raw_view_masks: binarized raw masks B X C X H X W 
        '''
        _,_,h,mask_dim = raw_view_masks.shape
        #breakpoint()
        assert mask_dim == h
        idx = torch.LongTensor([1,0,3,2]).cuda()
        rois_1 = [roi_t[j,:1,:4].index_select(-1, idx)*mask_dim for j in range(roi_t.shape[0])]
        rois_2 = [roi_t[j,1:2,:4].index_select(-1, idx)*mask_dim for j in range(roi_t.shape[0])]
        flip_1 = roi_t[:,0,4]
        flip_2 = roi_t[:,1,4]
        #detached_mask = masks.detach()
        aligned_1 = self.handle_flip(ops.roi_align(raw_view_masks,rois_1,56),flip_1) # mask output is B X 16 X 7 X 7
        aligned_2 = self.handle_flip(ops.roi_align(raw_view_masks,rois_2,56),flip_2) # mask output is B X 16 X 7 X 7
        return aligned_1,aligned_2

    def sample_mask_by_area(self,mask1,mask2,mask_area_1,mask_area_2,min_area,max_area,n_masks,resize_to=None):
        mask_in_range = torch.logical_and(torch.greater(mask_area_1, min_area),torch.less_equal(mask_area_1, max_area))
        mask_in_range *= torch.logical_and(torch.greater(mask_area_2, min_area),torch.less_equal(mask_area_2, max_area))
        sel_masks = mask_in_range.float() + 0.00000000001
        sel_masks = sel_masks / sel_masks.sum(1, keepdims=True)
        sel_masks = torch.log(sel_masks)
        dist = torch.distributions.categorical.Categorical(logits=sel_masks)
        mask_ids = dist.sample([n_masks]).T
        b = mask1.shape[0]
        mask1 = torch.stack([mask1[b][mask_ids[b]] for b in range(b)])
        mask2  = torch.stack([mask2[b][mask_ids[b]] for b in range(b)])

        if resize_to:
            mask1 = F.interpolate(mask1,resize_to)
            mask2 = F.interpolate(mask2,resize_to)
        return mask1,mask2,mask_ids

    def _joint_sample_masks(self,view1_mask,view2_mask,n_masks=16):
        '''
        view1_mask,view2_mask: B X C X H X W 
        '''
        MIN_C5,MAX_C5 = (0.05,0.90)
        MIN_C4,MAX_C4 = (0.02,0.5)
        MIN_C3,MAX_C3 = (0.01,0.15)
        b,c,h,w = view1_mask.shape
        mask_area_1 = view1_mask.sum((-1,-2))
        mask_area_2 = view2_mask.sum((-1,-2))
        mask_exists = torch.logical_and(torch.greater(mask_area_1, 1e-3),torch.greater(mask_area_2, 1e-3))
        mask_area_1 /= (h*w) # convert to percentage
        mask_area_2 /= (h*w)
        # Calc valid
        mask1_c5, mask2_c5,mask_id_c5 = self.sample_mask_by_area(view1_mask,view2_mask,mask_area_1,mask_area_2,MIN_C5,MAX_C5,n_masks,resize_to=(7,7))
        mask1_c4, mask2_c4,mask_id_c4 = self.sample_mask_by_area(view1_mask,view2_mask,mask_area_1,mask_area_2,MIN_C4,MAX_C4,n_masks,resize_to=(14,14))
        mask1_c3, mask2_c3,mask_id_c3 = self.sample_mask_by_area(view1_mask,view2_mask,mask_area_1,mask_area_2,MIN_C3,MAX_C3,n_masks,resize_to=(28,28))
        return mask1_c5,mask2_c5,mask1_c4,mask2_c4,mask1_c3,mask2_c3

    def joint_sample_masks(self,view1_mask,view2_mask,n_masks=16):
        '''
        # directly assign by layers
        view1_mask,view2_mask: B X C X H X W 
        '''
        b,cc,h,w = view1_mask.shape
        ca = cc//4
        view1_mask = view1_mask.view(b,4,ca,h,w)
        view2_mask = view2_mask.view(b,4,ca,h,w)
        mask1_c5 = view1_mask[:,2,...]
        mask1_c4 = view1_mask[:,1,...]
        mask1_c3 = view1_mask[:,0,...]
        mask2_c5 = view2_mask[:,2,...]
        mask2_c4 = view2_mask[:,1,...]
        mask2_c3 = view2_mask[:,0,...]
        # Calc valid
        return mask1_c5,mask2_c5,mask1_c4,mask2_c4,mask1_c3,mask2_c3
        
    def forward(self, view1, view2, mm, input_masks,raw_image,roi_t,slic_mask,user_masknet=False,full_view_prior_mask=None):
        # online network forward
        #import ipdb;ipdb.set_trace()
        #breakpoint()
        im_size = view1.shape[-1]
        assert im_size == 224
        with torch.no_grad():
            full_view_prior_mask = full_view_prior_mask[:,[0,2,3,4],:,:]
            masks = self.binarize_full_view_masks(full_view_prior_mask) # B X 5 X H X W ->  B X 5C X H X W 
            aligned_1,aligned_2 = self.roi_transforms(roi_t,masks.cuda())
            mask1_c5,mask2_c5,mask1_c4,mask2_c4,mask1_c3,mask2_c3 = self.joint_sample_masks(aligned_1,aligned_2,n_masks=16)
        c5_proj,c4_proj,c3_proj = self.online_network(
            torch.cat([view1, view2], dim=0),
            torch.cat([mask1_c5, mask2_c5], dim=0),
            torch.cat([mask1_c4, mask2_c4], dim=0),
            torch.cat([mask1_c3, mask2_c3], dim=0),
        )
        c5_target,c4_target,c3_target = self.target_network(
            torch.cat([view2, view1], dim=0),
            torch.cat([mask2_c5, mask1_c5], dim=0),
            torch.cat([mask2_c4, mask1_c4], dim=0),
            torch.cat([mask2_c3, mask1_c3], dim=0),
        )
        pred = self.predictor(torch.cat([c5_proj,c4_proj,c3_proj],dim=1))
        target = torch.cat([c5_target,c4_target,c3_target],dim=1)
        #breakpoint()
        out_masks = [torch.argmax(x[0],0).detach().cpu() for x in (mask1_c5,mask2_c5,
            mask1_c4,mask2_c4,
            mask1_c3,mask2_c3,)]+[
                x.detach().cpu() for x in
                [full_view_prior_mask[0,2,...],full_view_prior_mask[0,1,...],full_view_prior_mask[0,0,...]]]
        #breakpoint()
        out_imgs = [view1[0].permute(1,2,0),view2[0].permute(1,2,0),raw_image[0].permute(1,2,0)]
        out_imgs = [np.exp(x.detach().cpu()) for x in out_imgs]
        return pred,target,out_imgs,out_masks,[mask1_c5,mask2_c5,mask1_c4,mask2_c4,mask1_c3,mask2_c3]

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
