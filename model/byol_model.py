#-*- coding:utf-8 -*-
# Nearest neighbor code from https://github.com/vturrisi/solo-learn/blob/main/solo/methods/nnbyol.py
import torch
from .basic_modules import EncoderwithProjection, Predictor, Masknet
#from utils.mask_utils import convert_binary_mask
from torchvision import ops
# from utils.visualize_masks import wandb_set

class BYOLModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pool_size = config['loss']['pool_size']
        self.train_batch_size = config['data']['train_batch_size']
        self.encoder_output_dim = config['model']['projection']['input_dim']
        self.projection_dim= config['model']['projection']['output_dim']

        # online network
        self.online_network = EncoderwithProjection(config)

        # target network
        self.target_network = EncoderwithProjection(config)
        
        #mask net
        self.masknet = Masknet(self.target_network.encoder, self.target_network.projetion, config)
        self.spatial_softmax = torch.nn.Softmax2d()
        
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

    def forward(self, view0, view1, view2, mm, wandb_id, roi_tranform):
        # online network forward
        
        #Get mask for whole image (B, 16, 14, 14)
        b = view0.size(0)
        projected_view0 = self.target_network.projetion(
                                                        torch.reshape(self.target_network.encoder(view0), (b, -1, self.encoder_output_dim))
                                                        ) #(B,49,256)
        projected_view0 = torch.reshape(projected_view0, (b, self.pool_size, self.pool_size, self.projection_dim)).permute(0,3,1,2) #(B, 256, Pool, Pool)
        masks, mask_ids = self.masknet(view0) #(B, 16, Pool Pool)
        # masks = torch.randn((view0.size(0), 16, 14, 14)).cuda()
        # mask_ids = torch.arange(0, 16).expand(2*view0.size(0),-1).cuda()

        _,_,_,mask_dim = masks.shape
        idx = torch.LongTensor([1,0,3,2]).cuda()
        rois_1 = [roi_tranform[j,:1,:4].index_select(-1, idx)*mask_dim for j in range(roi_tranform.shape[0])]
        rois_2 = [roi_tranform[j,1:2,:4].index_select(-1, idx)*mask_dim for j in range(roi_tranform.shape[0])]
        flip_1 = roi_tranform[:,0,4]
        flip_2 = roi_tranform[:,1,4]
        #TODO: SPATIAL SCALE??
        aligned_1 = self.handle_flip(ops.roi_align(masks,rois_1,7),flip_1) # mask output is B X 16 X 7 X 7
        aligned_2 = self.handle_flip(ops.roi_align(masks,rois_2,7),flip_2) # mask output is B X 16 X 7 X 7

        aligned_1 = self.spatial_softmax(aligned_1)
        aligned_2 = self.spatial_softmax(aligned_2)
        
        q,pinds = self.predictor(*self.online_network(torch.cat([view1, view2], dim=0),
                                                    torch.cat([aligned_1, aligned_2]).cuda(),
                                                    mask_ids,
                                                    wandb_id)
                                )

        # target network forward
        with torch.no_grad():
            self._update_target_network(mm)
            target_z, tinds = self.target_network(torch.cat([view2, view1], dim=0),
                                                torch.cat([aligned_2, aligned_1]).cuda(),
                                                mask_ids,
                                                wandb_id)
            target_z = target_z.detach().clone()

        return q, target_z, pinds, tinds, masks
