import random
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import inference_context,COCOEvaluator
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer,launch
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
#from trainer import *
import numpy as np
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper, build_detection_test_loader, build_detection_train_loader
#longest side transform 

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


class EvalPadding(T.Augmentation):
    def __init__(self,target_size=1024):
        self.target_size = target_size
        super().__init__()

    def get_transform(self, image):
        old_h, old_w = image.shape[:2]
        assert old_h == self.target_size or old_w == self.target_size
        delta_a = self.target_size - old_h
        delta_b = self.target_size - old_w
        return T.PadTransform(old_h, old_w, new_h, new_w)

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
                        LongestSideRandomScale(1, 1, 1024), #During testing, images are resized to 1024 pixels on the longest side then padded to 1024×1024 pixels.
                        #T.FixedSizeCrop((1024, 1024)) #somewhat buggy, so remove now
                    ]
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, True, augmentations=test_augs, use_instance_mask=True))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def main_train():
    #torch.multiprocessing.freeze_support()
    cfg = get_cfg()
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file("./config_moco_200.yaml")
    trainer = Trainer(cfg) 
    trainer.resume_or_load(resume=True)
    return trainer.train()
    
if __name__ == '__main__':
    launch(main_train,8,num_machines=1, machine_rank=0, dist_url=None)
