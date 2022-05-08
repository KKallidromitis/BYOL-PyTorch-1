import random
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import inference_context,COCOEvaluator
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import Boxes
from detectron2 import model_zoo
import os


from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer,launch
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper, build_detection_test_loader, build_detection_train_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
#val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
import logging
logging.basicConfig(level=logging.INFO)
from train import Trainer


class NoPostProcessingModule(torch.nn.Module):

    def __init__(self,backbone):
        super().__init__()
        self.backbone = backbone
    
    def forward(self,x):
        inverses = []
        for input_dict in x:
            height = input_dict.pop('height')
            width = input_dict.pop('width')
            if height >= width:
                new_height, new_width = 1024, int(1024* width / height)
            else:
                new_width, new_height = 1024, int(1024 * height / width)
            augs = [
                T.CropTransform(0,0,new_width,new_height), #invert padding
                T.ScaleTransform(new_height,new_width,height,width,'bilinear')
            ]
            trans = T.TransformList(augs)
            inverses.append(trans)
        results = self.backbone.inference(x)

        #restore everything
        for result,inverse_transform in zip(results,inverses):
            instances = result['instances']
            boxes = instances.pred_boxes.tensor.cpu().numpy().astype(int)
            masks = instances.pred_masks
            masks = torch.permute(masks,(1,2,0)).cpu().numpy().astype(float)
            if masks.shape[-1] == 0:
                continue
            masks = inverse_transform.apply_image(masks)
            masks = masks >= 0.5
            masks = torch.BoolTensor(masks)
            masks = torch.permute(masks,(2,0,1))
            instances.pred_masks = masks
            boxes = boxes.reshape(-1,2)
            boxes = inverse_transform.apply_coords(boxes)
            boxes = boxes.reshape(-1,4)
            boxes = Boxes(torch.Tensor(boxes))
            instances.pred_boxes = boxes
            #print(masks.shape,type(masks),masks.dtype,torch.max(masks),torch.min(masks)) #sanity check
            #exit()
        return results

def main_train():
    #torch.multiprocessing.freeze_support()
    cfg = get_cfg()
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file("./config.yaml")
    #trainer = Trainer(cfg) 
    cfg.MODEL.WEIGHTS = 'outputbyol/model_final.pth'
    #trainer.resume_or_load(resume=True)
    predictor = DefaultPredictor(cfg)
    val_loader = Trainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])
    #val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    evaluater = COCOEvaluator(cfg.DATASETS.TEST[0], ["bbox","segm"],True,'./inference_byol')#
    model = predictor.model
    #model = NoPostProcessingModule(model)
    print(inference_on_dataset(model, val_loader, evaluater))
    return 0

main_train()
