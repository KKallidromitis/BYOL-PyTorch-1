from joblib import Parallel, delayed
from skimage.segmentation import felzenszwalb
import numpy as np
import pickle
import torchvision
import torch
import os
import time
import logging
from torchvision import datasets
import matplotlib.image as mpimg
import cv2 #For binanry mask edge detection
import argparse
from tqdm import tqdm

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    
    https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
    
    
class Preload_Masks():
    def __init__(self,dataset_dir,output_dir,ground_mask_dir='',mask_type='fh',experiment_name='',
                 num_threads=os.cpu_count(),scale=1000,min_size=1000,segments=[3,3]):
        
        self.output_dir=output_dir
        self.mask_type=mask_type
        self.scale = scale
        self.min_size = min_size
        self.segments = segments
        self.experiment_name = experiment_name
        self.num_threads = num_threads
        self.ground_mask_dir = ground_mask_dir
        self.totensor = torchvision.transforms.ToTensor()
        self.image_dataset = ImageFolderWithPaths(dataset_dir,transform=self.totensor)
        self.ds_length = len(self.image_dataset)
        self.save_path = os.path.join(self.output_dir,self.experiment_name)
        
    def create_patch_mask(self,image,segments):
        dims=list(np.floor_divide(image.shape[1:],segments))

        mask=torch.hstack([torch.cat([torch.zeros(dims[0],dims[1])+i+(j*(segments[0])) 
                                      for i in range(segments[0])]) for j in range(segments[1])])

        mods = list(np.mod(image.shape[1:],segments))
        if mods[0]!=0:
            mask = torch.cat([mask,torch.stack([mask[-1,:] for i in range(mods[0])])])
        if mods[1]!=0:
            mask = torch.hstack([mask,torch.stack([mask[:,-1] for i in range(mods[1])]).T])

        return mask.int()

    def create_fh_mask(self,image, scale, min_size):
        mask = felzenszwalb(image.permute(1,2,0), scale=scale, min_size=min_size)
        return torch.tensor(mask).int()
    
    def load_ground_mask(self,img_path):
        #assuming masks have ssame name ass images and asre png files
        mask_path = os.path.join(self.ground_mask_dir,os.path.splitext(img_path.split('/')[-1])[0]+'.png')
        mask = torch.tensor(mpimg.imread(mask_path)[:,:,0])
        return mask
    
    def select_mask(self,obj):
        image,label,img_path = obj
        suffix = '_'+self.mask_type+'.pkl'
        name = os.path.join(self.save_path,os.path.splitext('_'.join(img_path.split('/')[-2:]))[0])
        
        if self.mask_type =='fh':
            mask = self.create_fh_mask(image, scale=self.scale, min_size=self.min_size).to(dtype=torch.int16)
        if self.mask_type =='patch':  
            mask = self.create_patch_mask(image,segments=self.segments).to(dtype=torch.int16)
        if self.mask_type =='ground':
            mask = self.load_ground_mask(img_path).to(dtype=torch.int16)
        
        with open(name+suffix, 'wb') as handle:
            pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return [img_path,name+suffix]
    
    def pkl_save(self,file,name):
        with open(name, 'wb') as handle:
            pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def save_dicts(self,img_paths,mask_paths):
        self.pkl_save(mask_paths,os.path.join(self.output_dir,self.experiment_name+'_img_to_'+self.mask_type+'.pkl'))
        return
    
    def forward(self):
        try:
            os.mkdir(os.path.join(self.output_dir,self.experiment_name))
        except:
            if not os.path.exists(self.output_dir):
                os.makedirs(os.path.join(self.output_dir,self.experiment_name))
                
        print('Dataset Length: %d  '%(self.ds_length))
        start = time.time()
        img_paths,mask_paths = zip(*Parallel(n_jobs=self.num_threads,prefer="threads")
                                 (delayed(self.select_mask)(obj) for obj in tqdm(self.image_dataset)))
        end = time.time()

        self.save_dicts(img_paths,mask_paths)

        print('Time Taken: %f  '%((end - start)/60))
        
        return 
    
if __name__=="__main__":
    
    mask_loader = Preload_Masks(dataset_dir = '/home/kkallidromitis/data/imagenet/train',
                            output_dir = '/home/kkallidromitis/data/imagenet/masks/',
                            mask_type = 'fh',
                            experiment_name = 'train',
                            )
    mask_loader.forward()
