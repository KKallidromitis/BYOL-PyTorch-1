from byol_transform import *
import numpy as np
anno = '/home/jacklishufan/ByteTrack/datasets/coco/annotations/instances_train2017.json'
root = '/home/jacklishufan/ByteTrack/datasets/coco/train2017'
dataset = COCOMaskDataset(root,anno,None)
img,anno = dataset[0]
print(img,np.unique(anno))