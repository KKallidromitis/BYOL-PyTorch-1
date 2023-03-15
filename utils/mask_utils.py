import numpy as np
import torch
import skimage
import pickle
import torchvision
import torch.nn.functional as F
from torchvision import ops

def create_patch_mask(image,segments=[3,2]):
    totensor = torchvision.transforms.ToTensor()
    """
    Input is a PIL Image or Tensor with [CxWxH]
    """
    try:
        image = torchvision.transforms.ToTensor()(image)
    except:
        pass
    dims=list(np.floor_divide(image.shape[1:],segments))
    
    mask=torch.hstack([torch.cat([torch.zeros(dims[0],dims[1])+i+(j*(segments[0])) 
                                  for i in range(segments[0])]) for j in range(segments[1])])
    
    mods = list(np.mod(image.shape[1:],segments))
    if mods[0]!=0:
        mask = torch.cat([mask,torch.stack([mask[-1,:] for i in range(mods[0])])])
    if mods[1]!=0:
        mask = torch.hstack([mask,torch.stack([mask[:,-1] for i in range(mods[1])]).T])
        
    return mask.int()

def convert_binary_mask(mask,max_mask_id=257,pool_size=7):
    #breakpoint()
    batch_size = mask.shape[0]
    max_mask_id = max(max_mask_id,torch.max(mask).item())
    mask_ids = torch.arange(max_mask_id).reshape(1,max_mask_id, 1, 1).float().to('cuda')
    binary_mask = torch.eq(mask_ids, mask).float() # 64, 256, 224, 224]
    binary_mask = torch.nn.AdaptiveAvgPool2d((pool_size,pool_size))(binary_mask) # 64, 256, 7, 7
    binary_mask = binary_mask.reshape(batch_size,max_mask_id,pool_size*pool_size)
    #breakpoint()
    return binary_mask
    binary_mask = torch.reshape(binary_mask,(batch_size,max_mask_id,pool_size*pool_size)).permute(0,2,1) #64,49,256
    binary_mask = torch.argmax(binary_mask, axis=-1)
    binary_mask = torch.eye(max_mask_id)[binary_mask]
    binary_mask = binary_mask.permute(0, 2, 1)
    return binary_mask

def sample_masks(binary_mask,n_masks=16):
    batch_size=binary_mask.shape[0]
    #breakpoint()
    mask_exists = torch.greater(binary_mask.sum(-1), 1e-3)
    mask_exists = (mask_exists[:batch_size//2].float()+mask_exists[batch_size//2:].float() )>=2
    sel_masks = mask_exists.float() + 0.00000000001
    #breakpoint()
    sel_masks = sel_masks / sel_masks.sum(1, keepdims=True)
    sel_masks = torch.log(sel_masks)
    sel_masks = sel_masks[:,1:] # Do not sample channel0==Background
    
    dist = torch.distributions.categorical.Categorical(logits=sel_masks)
    mask_ids = dist.sample([n_masks]).T
    mask_ids = mask_ids.repeat([2,1])
    #breakpoint()
    #mask_ids[32:]=mask_ids[:32]
    mask_ids += 1 # never sample background
    sample_mask = torch.stack([binary_mask[b][mask_ids[b]] for b in range(batch_size)])
    
    return sample_mask,mask_ids

def maskpool(mask,x):
    '''
    mask: B X C_M X  H X W (1 hot encoding of mask)
    x: B X C X H X W (normal mask)
    '''
    _,c_m,_,_ = mask.shape
    b,c,h,w = x.shape
    mask = mask.view(b,c_m,h*w)
    mask_area = mask.sum(dim=-1,keepdims=True)
    mask = mask / torch.maximum(mask_area, torch.ones_like(mask))
    x = x.permute(0,2,3,1).reshape(b,h*w,c) # B X HW X C
    x = torch.matmul(mask.view(b,c_m,h*w).to('cuda'), x)
    return x,mask

def to_binary_mask(label_map,c_m=-1,resize_to=None):
    b,h,w = label_map.shape
    if c_m==-1:
        c_m = torch.max(label_map).item()+1
    label_map_one_hot = F.one_hot(label_map,c_m).permute(0,3,1,2).float()
    if resize_to is not None:
        label_map_one_hot = F.interpolate(label_map_one_hot,resize_to, mode='bilinear',align_corners=False)
        label_map = torch.argmax(label_map_one_hot,1)
        h,w = resize_to
    label_map_one_hot = label_map_one_hot.reshape(b,c_m,h*w)
    return label_map_one_hot.view(b,c_m,h,w)


def refine_mask(src_label,target_label,mask_dim,src_dim=16):
        # B X H_mask X W_mask
        n_tgt = torch.max(target_label).item()+1
        slic_mask = to_binary_mask(target_label,n_tgt,(mask_dim,mask_dim))  # B X 100 X H_mask X W_mask
        masknet_label_map = to_binary_mask(src_label,src_dim,(mask_dim,mask_dim)) # binary B X 16 X H_mask X W_mask
        pooled,_ =maskpool(slic_mask,masknet_label_map) # B X NUM_SLIC X N_MASKS
        pooled_ids = torch.argmax(pooled,-1) # B X NUM_SLIC  X 1 => label map
        converted_idx = torch.einsum('bchw,bc->bchw',slic_mask ,pooled_ids).sum(1).long().detach() #label map in hw space
        return converted_idx

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
    return aligned_1,aligned_2,(mask_aligned1*mask_aligned2 > 0.1).float()