import numpy as np
import torch
import skimage
import pickle
import torchvision
import torch.nn.functional as F

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

def to_binary_mask(label_map,c_m,resize_to=None):
    b,h,w = label_map.shape
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