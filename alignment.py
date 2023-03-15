from data.byol_transform import *
import numpy as np
from model import BYOLModel
import yaml
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch import nn
from skimage.segmentation import slic
from torchvision import transforms as TF
import kornia
from torchvision.models import resnet50
from utils.mask_utils import to_binary_mask,maskpool
def build_dataset():
    stage = 'train'
    mask_type = 'coco'
    ROOT = '/shared/group/coco/'
    image_dir = os.path.join(ROOT, f"{'train2017' if stage in ('train', 'ft') else 'val2017'}")
    annoFile = os.path.join(ROOT,'annotations', f"{'instances_train2017.json' if stage in ('train', 'ft') else 'instances_val2017.json'}")
    #mask_file = os.path.join(self.image_dir,'masks',stage+'_tf_img_to_'+self.mask_type+'.pkl')
    transform1 = get_transform(stage)
    transform2 = get_transform(stage, gb_prob=0.1, solarize_prob=0.2)
    transform3 = get_transform('raw')
    transform = MultiViewDataInjector([transform1, transform2,transform3],False,slic_segments=100)
    dataset = COCOMaskDataset(image_dir,annoFile,transform)
def handle_flip(aligned_mask,flip):
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
def inverse_align(features_1,features_2,roi_t,out_dim=56):
    '''
    features: N X C X H X W, view 1 & view 2
    '''
    n,c,h,w = features_1.shape
    roi_t_2 = roi_t.clone()
    dh =  (roi_t[...,2]-roi_t[...,0])
    dw =  (roi_t[...,3]-roi_t[...,1])
    roi_t_2[...,0] = (0.0 - roi_t[...,0]) / dh
    roi_t_2[...,1] = (0.0 - roi_t[...,1]) / dw
    roi_t_2[...,2] = (1.0 - roi_t[...,0]) / dh
    roi_t_2[...,3] = (1.0 - roi_t[...,1]) / dw
    idx = torch.LongTensor([1,0,3,2]).to(roi_t.device)
    mask_dim = features_1.shape[-1]
    rois_1 = [roi_t_2[j,:1,:4].index_select(-1, idx)*mask_dim for j in range(roi_t.shape[0])]
    rois_2 = [roi_t_2[j,1:2,:4].index_select(-1, idx)*mask_dim for j in range(roi_t.shape[0])]
    flip_1 = roi_t[:,0,4]
    flip_2 = roi_t[:,1,4]
    assert features_1.shape == features_2.shape
    aligned_1 = ops.roi_align(handle_flip(features_1,flip_1),rois_1,out_dim)
    aligned_2 = ops.roi_align(handle_flip(features_2,flip_2),rois_2,out_dim)
    mask = torch.ones(n,1,h,w).to(features_1.device).float()
    mask_aligned1 = ops.roi_align(handle_flip(mask,flip_1),rois_1,out_dim)
    mask_aligned2 = ops.roi_align(handle_flip(mask,flip_2),rois_2,out_dim)
    return aligned_1,aligned_2,(mask_aligned1*mask_aligned2 > 0).float()

def main(args):
    dataset = build_dataset()
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=4)
    model = resnet50(pretrained=None)
    state = torch.load(args.checkpoint)
    
    r = model.load_state_dict(state,strict=False)
    model = nn.Sequential(*list(model.children())[:-2])
    print(r)
    device = 'cuda'
    model.eval()
    model.to(device)
    for images,labels,roi_t in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        roi_t = roi_t.to(device)
        view1 = images[:,0]
        view2 = images[:,1]
        feature_1 = model(view1)
        feature_1 = F.normalize(feature_1,dim=-1)
        feature_2 = model(view2)
        feature_2 = F.normalize(feature_2,dim=-1)
        feature_1,feature_2,msk_overlap = inverse_align(feature_1,feature_2,roi_t,224)
        mask_intersect_object = labels[:,2,0] * msk_overlap[:,0].long()
        gt_mask = to_binary_mask(mask_intersect_object.long())
        feature_1_pooled = maskpool(gt_mask,feature_1)
        feature_2_pooled = maskpool(gt_mask,feature_2)