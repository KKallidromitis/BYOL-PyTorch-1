#-*- coding:utf-8 -*-
import torch
from .basic_modules import EncoderwithProjection, Predictor, Masknet, SpatialAttentionMasknet
from utils.mask_utils import convert_binary_mask,sample_masks
from torchvision import ops
class BYOLModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # online network
        self.online_network = EncoderwithProjection(config)

        # target network
        self.target_network = EncoderwithProjection(config)
        
        #mask net
        self.masknet = SpatialAttentionMasknet()
        
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

    def _new_forward(self, view1, view2, mm, masks,raw_image,roi_t):
        # online network forward
        #import ipdb;ipdb.set_trace()
        #breakpoint()
        im_size = view1.shape[-1]
        assert im_size == 224
        idx = torch.LongTensor([1,0,3,2])
        rois_1 = [roi_t[j,:1,:4].index_select(-1, idx)*im_size for j in range(roi_t.shape[0])]
        rois_2 = [roi_t[j,1:2,:4].index_select(-1, idx)*im_size for j in range(roi_t.shape[0])]
        masks = self.masknet(raw_image) # raw_image: B X 3 X H X W, ->Mask B X 16 X H_mask X W_mask
        aligned_1 = ops.roi_align(masks,rois_1,7) # mask output is B X 16 X 7 X 7
        aligned_2 = ops.roi_align(masks,rois_2,7) # mask output is B X 16 X 7 X 7
        q,pinds = self.predictor(*self.online_network(torch.cat([view1, view2], dim=0),torch.cat([aligned_1, aligned_2], dim=0),None,None))
        # target network forward
        with torch.no_grad():
            self._update_target_network(mm)
            target_z, tinds = self.target_network(torch.cat([view2, view1], dim=0),torch.cat([aligned_2,aligned_1]).to('cuda'),None,None)
            target_z = target_z.detach().clone()

        return q, target_z, pinds, tinds,masks

    def forward(self, view1, view2, mm, masks,raw_image,roi_t):
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
