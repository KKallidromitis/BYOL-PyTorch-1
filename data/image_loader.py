#-*- coding:utf-8 -*-
import torch
import os
from torchvision import datasets,transforms
from .byol_transform import MultiViewDataInjector, get_transform, SSLMaskDataset,COCOMaskDataset


class ImageLoader():
    def __init__(self, config):
        self.image_dir = config['data']['image_dir']
        self.num_replicas = config['world_size']
        self.rank = config['rank']
        self.distributed = config['distributed']
        self.resize_size = config['data']['resize_size']
        self.data_workers = config['data']['data_workers']
        self.dual_views = config['data']['dual_views']
        self.mask_type = config['data']['mask_type']
        self.over_lap_mask = config['data'].get('over_lap_mask',True)
        self.slic_segments = config['data']['slic_segments']
        self.subset = config['data'].get("subset", "")
        self.do_slic = not config['clustering']['no_slic']
        self.super_pixel = config['clustering']['superpixel']

    def get_loader(self, stage, batch_size):
        dataset = self.get_dataset(stage)
        if self.distributed and stage in ('train', 'ft'):
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=self.num_replicas, rank=self.rank)
        else:
            self.train_sampler = None

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None and stage not in ('val', 'test')),
            num_workers=self.data_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=True,
            persistent_workers=True
        )
        return data_loader

    def get_dataset(self, stage):

        image_dir = os.path.join(self.image_dir, "images", f"{'train' if stage in ('train', 'ft') else 'val'}")
        if self.mask_type == "":
            mask_file= None
            mask_file_path = None #If we don't need to load we can avoid
        else:
            mask_file = os.path.join(self.image_dir,'masks',stage+'_tf_img_to_'+self.mask_type+'.pkl')
            mask_file_path = os.path.join(self.image_dir,'masks','train_tf')
        
        transform1 = get_transform(stage,crop_size=self.resize_size)
        transform2 = get_transform(stage, gb_prob=0.1, solarize_prob=0.2,crop_size=self.resize_size)
        transform3 = get_transform('raw',crop_size=self.resize_size)

        transform = MultiViewDataInjector([transform1, transform2,transform3],self.over_lap_mask,slic_segments=self.slic_segments,do_slic=self.do_slic)
        
        dataset = SSLMaskDataset(image_dir,mask_file,transform=transform,mask_file_path=mask_file_path, subset=self.subset)
        return dataset

    def set_epoch(self, epoch):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)


class ImageLoadeCOCO():
    def __init__(self, config):
        self.image_dir = config['data']['image_dir']
        self.num_replicas = config['world_size']
        self.rank = config['rank']
        self.distributed = config['distributed']
        self.resize_size = config['data']['resize_size']
        self.data_workers = config['data']['data_workers']
        self.dual_views = config['data']['dual_views']
        self.mask_type = config['data']['mask_type']
        self.slic_segments = config['data']['slic_segments']
        self.over_lap_mask = config['data'].get('over_lap_mask',True)

    def get_loader(self, stage, batch_size):
        dataset = self.get_dataset(stage)
        if self.distributed and stage in ('train', 'ft'):
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=self.num_replicas, rank=self.rank)
        else:
            self.train_sampler = None

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None and stage not in ('val', 'test')),
            num_workers=self.data_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=True
        )
        return data_loader

    def get_dataset(self, stage):
        #import ipdb;ipdb.set_trace()
        image_dir = os.path.join(self.image_dir, f"{'train2017' if stage in ('train', 'ft') else 'val2017'}")
        annoFile = os.path.join(self.image_dir,'annotations', f"{'instances_train2017.json' if stage in ('train', 'ft') else 'instances_val2017.json'}")
        #mask_file = os.path.join(self.image_dir,'masks',stage+'_tf_img_to_'+self.mask_type+'.pkl')
        transform1 = get_transform(stage)
        transform2 = get_transform(stage, gb_prob=0.1, solarize_prob=0.2)
        transform3 = get_transform('raw')
        transform = MultiViewDataInjector([transform1, transform2,transform3],self.over_lap_mask,slic_segments=self.slic_segments )
        dataset = COCOMaskDataset(image_dir,annoFile,transform)
        return dataset

    def set_epoch(self, epoch):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
