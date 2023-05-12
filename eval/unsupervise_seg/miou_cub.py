from torchvision import models
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from mmseg.datasets import build_dataset
from matplotlib import pyplot as plt
import torchmetrics
import numpy as np
from sklearn.cluster import KMeans,MiniBatchKMeans,SpectralClustering
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import tqdm
from torch.utils.data import Subset
# DATASET PATH

from cub_dataset import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset',default='cub',type=str)
parser.add_argument('-m','--model',default='no_slic',type=str)
args = parser.parse_args()
if args.dataset == 'cub':
    path = '/shared/jacklishufan/CUB/CUB_200_2011/images'
    maske_math = '/shared/jacklishufan/CUB/segmentations'
    dataset = CUBDataset(path,maske_math)
elif args.dataset == 'flower':
    path = '/shared/jacklishufan/data/flowers/jpg'
    maske_math = '/shared/jacklishufan/data/flowers/segmim'
    dataset = FlowerDataset(path,maske_math)
    r = dataset[0]
else:
    raise NotImplemented

N_CLUSTERS = 4

kmeans = KMeans(n_clusters=N_CLUSTERS,n_init=1)
backbone = models.resnet50(pretrained=False)

# MODEL PATH 

CKPT = dict(r2o = '/shared/jacklishufan/05-18-01-49-300.pth',
pixpro = '/shared/jacklishufan/pixpro-400.pth',
# n_718 = '/shared/jacklishufan/07-18-16-25.pth',
# n1105 = '/shared/jacklishufan/ablation2/11-05-03-13.pth',
regioncl = '/shared/jacklishufan/mmcls/region-cl-200.pth',
dercon = '/shared/jacklishufan/detcoon-fullscale-bn-300.pth',
slotcon = '/home/jacklishufan/mmsegmentation/mmcls/slotcon_imagenet_r50_200ep.pth',
no_slic = '/home/jacklishufan/mmsegmentation/03-10-12-57-noslic.pth')

path = CKPT.get(args.model,args.model)

state = torch.load(path)#['state_dict']
if 'state_dict' in state:
    state = state['state_dict']
backbone.load_state_dict(state,strict=False)
encoder = nn.Sequential(*list(backbone.children())[:-2])
encoder = IntermediateLayerGetter(encoder, return_layers={'7':'out','6':'c4'})
# data_config=dict(
#         type='PascalVOCDataset',
#         data_root='data/VOCdevkit/VOC2012',
#         img_dir='JPEGImages',
#         ann_dir='SegmentationClass',
#         split=['ImageSets/Segmentation/val.txt',            
#               # 'ImageSets/Segmentation/train.txt',
#             #'ImageSets/Segmentation/aug.txt'
#             ],
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(type='LoadAnnotations'),
#             dict(
#                 type='MultiScaleFlipAug',
#                 #img_scale=(2048, 512),
#                 img_scale=None,
#                 img_ratios=[1.0],
#                 flip=False,
#                 transforms=[
#                     dict(type='Resize', keep_ratio=True),
#                     dict(type='RandomFlip'),
#                     dict(
#                         type='Normalize',
#                         mean=[123.675, 116.28, 103.53],
#                         std=[58.395, 57.12, 57.375],
#                         to_rgb=True),
#                     dict(type='DefaultFormatBundle'),
#                     dict(type='Collect', keys=['img', 'gt_semantic_seg'])
#                     #dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
#                 ])
#         ])
encoder.eval()


N_CLASS = 2
miou_calculator = torchmetrics.JaccardIndex(2)
cf_mat_calculator = torchmetrics.ConfusionMatrix(2)
cf_mat = torch.zeros(2,2)
preds = []
targets = []



def miou(preds,targets,cf_mat=None,count=None):
    if len(preds) >0:
        preds = torch.cat(preds)
        targets = torch.cat(targets)
    if cf_mat is not None:
        r = torchmetrics.functional.classification.jaccard._jaccard_from_confmat(cf_mat, N_CLASS,
                'macro',
                None,
                0.0,)
    else:
        miou_calculator = torchmetrics.JaccardIndex(2)
        r = miou_calculator(preds,targets)
    if count is not None:
        correct,acc_l = count
        acc = correct.float() / acc_l
    else:
        #iou = (ss >= 2).sum().float() / (ss>= 1).sum()
        acc = (preds==targets).sum().float() / len(preds.reshape(-1))
    return {"miou":r,"acc":acc}

N_SAMPLES = 1020
n_len = range(len(dataset))
samples = np.random.choice(n_len,N_SAMPLES,replace=False)
dataset = Subset(dataset,samples)
data_loader = DataLoader(dataset=dataset,shuffle=True,batch_size=1)
encoder.to('cuda:0')
idx = 0
acc = acc_l = 0
for img,gt_mask in tqdm.cli.tqdm(data_loader):
    #img  # 3 X H X W
    #gt_mask # 1 X H X W
    img = img[0]
    gt_mask = gt_mask[0]
    h=w=128 * 10
    img = img.to('cuda:0').unsqueeze(0)
    img = F.interpolate(img,(h,w),mode='bilinear',align_corners=False)
    idx += 1
    with torch.no_grad():
        features = encoder(img)['c4'][0].detach().cpu()
    c,h,w = features.shape
    #breakpoint()
    #features = F.interpolate(features.unsqueeze(0),(h,w),mode='bilinear')
    #breakpoint()
    features = (F.normalize(features.reshape(c,h*w).T,dim=-1)).detach().cpu()
    
    #upsample 
    kmeans.fit(features)

    labels = kmeans.labels_.reshape(h,w)
    labels = torch.LongTensor(labels)
    h=w=128
    labels = F.interpolate(F.one_hot(labels.unsqueeze(0)).permute(0,3,1,2).float(),(h,w),mode='bilinear',align_corners=False)
    labels = labels.argmax(dim=1)[0]
    #breakpoint()
    down_sample_binary_mask = F.interpolate(gt_mask.unsqueeze(0).float(),(h,w),mode='area')
    #down_sample_binary_mask[:,-1,:,:] = 0
    #down_sample_binary_mask[:,0,:,:] += 1e-6
    down_sample_binary_mask = (down_sample_binary_mask[0][0] > 0.5).long()
    unique_idx = np.unique(down_sample_binary_mask)
    #breakpoint()
    #huguarian matching
    labels_bin = F.one_hot(labels)
    if down_sample_binary_mask.max() < 1:
        continue
    down_sample_binary_mask_bin = F.one_hot(torch.LongTensor(down_sample_binary_mask))
    #breakpoint()
    intersection = torch.einsum('xya,xyb->ab',down_sample_binary_mask_bin,labels_bin)
    union = labels_bin.unsqueeze(2) + down_sample_binary_mask_bin.unsqueeze(3)
    union = (union > 0).float().sum(dim=(0,1))
    iou = intersection / union
    iou = iou[unique_idx,:] # num_unique_labels X K
    tgt_label = iou[1].argmax()
    #r,c = linear_sum_assignment(1-iou)
    #target_label = unique_idx[r]
    # final_label_map = torch.zeros_like(labels)
    # for src,tgt in zip(c,target_label):
    #     final_label_map += (labels == src).int() * tgt
    final_label_map = (labels == tgt_label).int()
    final_label_map = final_label_map.detach().cpu().reshape(-1)
    down_sample_binary_mask = down_sample_binary_mask.detach().cpu().reshape(-1)
    acc += (final_label_map == down_sample_binary_mask).sum()
    acc_l += len(final_label_map)
    # preds.append(final_label_map)
    # targets.append(down_sample_binary_mask)
    cf_mat += cf_mat_calculator(final_label_map,down_sample_binary_mask)
    
    if idx % 10 == 0:
        r = miou(preds,targets,cf_mat,(acc,acc_l))
        print(r,idx)
r = miou(preds,targets,cf_mat,(acc,acc_l))
print('-----------------------------------MIOU--------------------------------------')
print(r)