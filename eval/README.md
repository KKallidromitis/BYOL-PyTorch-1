
## Evaluation
Note: Instructions and code taken from the MoCo codebase. 

The `main_lincls.py` run linear evaluation on ImageNet following optmimzer in BYOL (Section C.1). 

The `detectron_train_net.py` script trains MaskRCNN (ResNet50-FPN) for 12 epochs on COCO 2017 for instance segmentation and object detection.

### ImageNet Instructions
Run the following. Note, BYOL used the best result following a sweep over lr in [0.4, 0.3, 0.2, 0.1, 0.05]
python main_lincls.py \
  -a resnet50 \
  --lr [LR] \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]


### Detectron2 Instructions

1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

2. Convert a pre-trained model to detectron2's format:
```
python3 convert-torchvision-to-d2.py [PATH-TO-PYTORCH-RESNET50-PTH-TAR] detcon_r50.pkl
```

3. Update config file (`/configs/COCO/mask_rcnn_R_50_FPN_1x.yaml`) with pkl path (`detcon_r50.pkl`).

4. Put COCO 2017 under "./datasets" directory,
following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
    requried by detectron2.

5. Run training:
    ```
    python detectron_train_net.py --config-file configs/COCO/mask_rcnn_R_50_FPN_1x.yaml \
        --num-gpus 8
    ```

### Results
TBD
