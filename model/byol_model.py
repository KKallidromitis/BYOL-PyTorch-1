#-*- coding:utf-8 -*-
import torch
from .basic_modules import EncoderwithProjection, Predictor, Masknet, SpatialAttentionMasknet,FCNMaskNet
from utils.mask_utils import convert_binary_mask,sample_masks
from torchvision import ops
import torch.nn.functional as F
class BYOLModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # online network
        self.online_network = EncoderwithProjection(config)

        # target network
        self.target_network = EncoderwithProjection(config)
        
        #mask net
        self.masknet = FCNMaskNet()
        
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

    @torch.no_grad()
    def _update_mask_network(self, mm):
        """Momentum update of maks network"""
        #TODO: set this up properly, now masknet just use online encoder
        for param_q, param_k in zip(self.online_network.encoder.parameters(), self.masknet.encoder.parameters()):
            param_k.data.mul_(mm).add_(1. - mm, param_q.data)

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


    def forward(self, view1, view2, mm, masks,raw_image,roi_t):
        # online network forward
        #import ipdb;ipdb.set_trace()
        #breakpoint()
        im_size = view1.shape[-1]
        assert im_size == 224
        idx = torch.LongTensor([1,0,3,2]).cuda()
        raw_encoding = self.target_network.encoder(raw_image) # B X 2048 X 7 X 7
        masks = self.masknet(raw_encoding) # raw_image: B X 3 X H X W, ->Mask B X 16 X H_mask X W_mask
        _,_,_,mask_dim = masks.shape
        rois_1 = [roi_t[j,:1,:4].index_select(-1, idx)*mask_dim for j in range(roi_t.shape[0])]
        rois_2 = [roi_t[j,1:2,:4].index_select(-1, idx)*mask_dim for j in range(roi_t.shape[0])]
        flip_1 = roi_t[:,0,4]
        flip_2 = roi_t[:,1,4]
        aligned_1 = self.handle_flip(ops.roi_align(masks,rois_1,7),flip_1) # mask output is B X 16 X 7 X 7
        aligned_2 = self.handle_flip(ops.roi_align(masks,rois_2,7),flip_2) # mask output is B X 16 X 7 X 7
        #breakpoint()
        #breakpoint()
        mask_b,mask_c,h,w =aligned_1.shape
        aligned_1 = aligned_1.reshape(mask_b,mask_c,h*w)
        aligned_2 = aligned_2.reshape(mask_b,mask_c,h*w)
        aligned_1 = F.softmax(aligned_1,dim=-2)
        aligned_2 = F.softmax(aligned_2,dim=-2)
        mask_ids = None
        masks = torch.cat([aligned_1, aligned_2])
        q,pinds = self.predictor(*self.online_network(torch.cat([view1, view2], dim=0),masks,mask_ids,mask_ids))
        # target network forward
        with torch.no_grad():
            self._update_target_network(mm)
            target_z, tinds = self.target_network(torch.cat([view2, view1], dim=0),torch.cat([aligned_2,aligned_1]).to('cuda'),mask_ids,mask_ids)
            target_z = target_z.detach().clone()

        return q, target_z, pinds, tinds,masks

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
