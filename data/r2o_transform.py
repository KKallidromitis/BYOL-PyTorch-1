#-*- coding:utf-8 -*-
import imp
import torch
from torchvision import transforms
import cv2
from PIL import Image, ImageOps
import numpy as np
import pickle
from torchvision.datasets import VisionDataset
from torchvision import ops
from torchvision.datasets.folder import default_loader,make_dataset,IMG_EXTENSIONS
from pycocotools.coco import COCO
import os
from skimage.segmentation import slic

def get_differentialble_transform(i,j,h,w,flip,crop_size):
        def g(x): # differentiable transform
            # B X C X H X W
            x = ops.roi_align(x,[ torch.FloatTensor([i,j,i+h,j+w]),],crop_size) # ROI Align == crop + resize
            if flip:
                x = torch.flip(x,dim=-1)
            return x
        return g

def to_slic(img,**kwargs):
    img = img.permute(1, 2, 0)
    h, w, c = img.size()
    seg = slic(img.to(torch.double).numpy(), start_label=0, **kwargs)
    seg = torch.from_numpy(seg)
    return seg.view(1, h, w)

class MultiViewDataInjector():
    def __init__(self, transform_list,over_lap_mask=True,flip_p=0.5,crop_size=224,slic_segments=100):
        self.transform_list = transform_list
        self.over_lap_mask = over_lap_mask
        self.crop_size = crop_size
        self.p = flip_p
        self.slic_segments = slic_segments
        self.slic = True

    def _get_crop_box(self,image):
        return transforms.RandomResizedCrop.get_params(image,scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0))

    def __call__(self,sample,mask):
        ww,hh = sample.size
        i1, j1, h1, w1 = self._get_crop_box(sample)
        i2, j2, h2, w2 = self._get_crop_box(sample)
        do_flip1 = torch.rand(1) < self.p
        do_flip2 = torch.rand(1) < self.p

        i_min = max(i1,i2)
        i_max = min(i1+h1,i2+h2)
        j_min = max(j1,j2)
        j_max = min(j1+w1,j2+w2)
        h = i_max-i_min
        w = j_max-j_min
        area = h * w if h > 0 and w > 0 else 0

        if self.over_lap_mask:
            intersect_masks = torch.zeros_like(mask)
            if area > 0:
                intersect_masks[:,i_min:i_max,j_min:j_max] = 1

            mask = intersect_masks * mask

        assert len(self.transform_list) == 3
        output0,mask0 = self.transform_list[0](sample,mask,(i1, j1, h1, w1),do_flip1)
        output1,mask1 = self.transform_list[1](sample,mask,(i2, j2, h2, w2),do_flip2)
        output2,mask2 = self.transform_list[2](sample,mask,(0, 0, hh, ww),False)
        mask0 = torch.cat([mask0,torch.ones_like(mask0[:1])])
        mask1 = torch.cat([mask1,torch.ones_like(mask1[:1])])
        if self.slic:
            super_pixel_id_map = to_slic(output2,n_segments=self.slic_segments) # SLIC GROUPING to 100 superpixels 1X H X W
            mask2 = torch.cat([mask2,super_pixel_id_map])
        else:
            mask2 = torch.cat([mask2,torch.ones_like(mask2[:1])])
        output_cat = torch.stack([output0,output1,output2], dim=0)

        #Hard code pipeline for generate mask for encoder
        transform1 = [i1,j1,i1+h1,j1+w1,do_flip1]
        transform2 = [i2,j2,i2+h2,j2+w2,do_flip2]
        transform3 = [0,0,hh,ww,0]
        transforms = torch.FloatTensor([transform1,transform2,transform3])
        mask_cat = torch.stack([mask0,mask1,mask2])
        transforms[:,[0,2]] /= hh
        transforms[:,[1,3]] /= ww
        return output_cat,mask_cat,transforms

class SSLMaskDataset(VisionDataset):
    def __init__(self, root: str, extensions = IMG_EXTENSIONS, transform = None, subset=""):
        self.root = root
        self.transform = transform
        if subset == "":
            self.samples = make_dataset(self.root, extensions = extensions,) #Pytorch 1.9+
        else:
            if subset not in ["imagenet1p", "imagenet100"]:
                raise NotImplementedError()
            
            if subset == "imagenet1p":
                with open('1percent.txt') as f:
                    samples = f.readlines()
                    samples = [x.replace('\n','').strip() for x in samples ]
                    samples = [x for x in samples if x]
                    samples = [(os.path.join(root,x.split('_')[0],x),None) for x in samples]
                    #samples = [x for x in samples if os.path.exists(x[0])]
                    self.samples = sorted(samples)
                    
            elif subset == "imagenet100":
                with open("imagenet100.txt") as f:
                    subset_classes = f.read().splitlines()
                assert len(subset_classes) == 100
                samples = []
                for c in subset_classes:
                    samples.extend([(os.path.join(root, c, f), None) for f in os.listdir(os.path.join(root, c))])
                self.samples = samples

        self.loader = default_loader
        
    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        
        # Load Image
        sample = self.loader(path)
        
        # Apply transforms
        if self.transform is not None:
            sample,diff_transfrom = self.transform(sample)
        return sample,diff_transfrom

    def __len__(self) -> int:
        return len(self.samples)
    
class GaussianBlur():
    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = cv2.GaussianBlur(np.array(img), (self.kernel_size, self.kernel_size), sigma)
        return Image.fromarray(img.astype(np.uint8))
    
class Solarize():
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, sample):
        return ImageOps.solarize(sample, self.threshold)

def get_transform(stage, gb_prob=1.0, solarize_prob=0., crop_size=224,crop_cordinates=None):
    #i, j, h, w = crop_cordinates
    t_list = []
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if stage in ('train', 'val'):
        t_list = [
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(kernel_size=23)], p=gb_prob),
            transforms.RandomApply([Solarize()], p=solarize_prob),
            transforms.ToTensor(),
            normalize]
        
    elif stage == 'raw':
        t_list = [
            transforms.RandomResizedCrop(crop_size),
            transforms.ToTensor(),
            normalize]
        
    transform = transforms.Compose(t_list)
    return transform
