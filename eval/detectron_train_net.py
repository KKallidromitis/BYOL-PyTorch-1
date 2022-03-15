#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import os
import torch
import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator

from detectron2.data import DatasetMapper, build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

class LongestSideRandomScale(T.Augmentation):

    def __init__(self,min_scale,max_scale,longest_side_length):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.longest_side_length = longest_side_length
        super().__init__()

    def get_transform(self, image):
        if self.min_scale == self.max_scale:
            r =  self.min_scale
        else:
            r = np.random.uniform(self.min_scale,self.max_scale )
        target = int(r * self.longest_side_length)
        old_h, old_w = image.shape[:2]
        old_long,old_short = max(old_h,old_w),min(old_h,old_w)
        new_long,new_short = target, int(target * old_short/old_long)
        if old_h >= old_w:
            new_h, new_w = new_long,new_short
        else:
            new_h, new_w = new_short,new_long
        return T.ResizeTransform(old_h, old_w, new_h, new_w)

class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        # TODO: Ask about detcon specific data aug (rn, using approximations)
        train_augs = [
                        T.RandomFlip(0.5),
                        LongestSideRandomScale(0.8, 1.25, 1024), #Approximating longest edge resizing by factor in range [0.85, 1.25]
                        T.FixedSizeCrop((1024, 1024)) #Then cropped or padded to a 1024×1024 image.
                    ]
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True, augmentations = train_augs, use_instance_mask=True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # TODO: Ask about detcon specific data aug (rn, using approximations)
        test_augs = [
                        LongestSideRandomScale(1, 1, 1024),
                        T.FixedSizeCrop((1024, 1024)) #During testing, images are resized to 1024 pixels on the longest side then padded to 1024×1024 pixels.
                    ]
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False, augmentations=test_augs, use_instance_mask=True))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
