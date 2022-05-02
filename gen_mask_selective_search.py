from data.byol_transform import *
import numpy as np
from model import BYOLModel
import yaml
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch import nn
from skimage.segmentation import slic
from torchvision.models._utils import IntermediateLayerGetter
from sklearn.cluster import *
import tqdm
import os
from tqdm.contrib.concurrent import process_map 

anno = '/home/jacklishufan/ByteTrack/datasets/coco/annotations/instances_train2017.json'
image_dir="/home/jacklishufan/detconb/imagenet"
root = '/home/jacklishufan/ByteTrack/datasets/coco/train2017'
stage = 'train'
mask_type = 'fh'
transform1 = get_transform(stage)
transform2 = get_transform(stage, gb_prob=0.1, solarize_prob=0.2)
transform3 = get_transform('raw')
transform = MultiViewDataInjector([transform1, transform2,transform3],False)
image_dir_t = os.path.join(image_dir,'images', f"{'train' if stage in ('train', 'ft') else 'val'}")
mask_file = os.path.join(image_dir,'masks',stage+'_tf_img_to_'+mask_type+'.pkl')
mask_file_path = os.path.join(image_dir,'masks','train_tf')
dataset = SSLMaskDataset(image_dir_t,mask_file,transform=transform,mask_file_path=mask_file_path,gen_mask=True)
mask_file_path = dataset.mask_file_path
length = len(dataset)
ROOT = '/shared/jacklishufan/mask-selective-search' # SAVE PATH

def process(idx):
    views,masks,transforms =  dataset[idx]
    file_name = dataset.samples[idx][0].split('/')[-1].split('.')[0]
    #breakpoint()
    save_target = masks[2].cpu().numpy()
    tgt = os.path.join(ROOT,file_name+'.pkl')
    with open(tgt, 'wb') as handle:
        pickle.dump(save_target,handle)


r = process_map(process, range(length), max_workers=16)
