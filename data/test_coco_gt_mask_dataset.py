from byol_transform import *
import numpy as np
anno = '/home/jacklishufan/ByteTrack/datasets/coco/annotations/instances_train2017.json'
root = '/home/jacklishufan/ByteTrack/datasets/coco/train2017'
stage = 'train'
transform1 = get_transform(stage)
transform2 = get_transform(stage, gb_prob=0.1, solarize_prob=0.2)
transform = MultiViewDataInjector([transform1, transform2])
dataset = COCOMaskDataset(root,anno,transform)
img,anno = dataset[0]
print(img,np.unique(anno))