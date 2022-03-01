
## Evaluation

The `linear_feature_eval.ipynb` is a Jupyter notebook which trains a linear classification head ontop of a ResNet backbone for ImageNet. All instructions for this evaluation are contained within the notebook.

The `detectron_train_net.py` script trains MaskRCNN (ResNet50-FPN) for 12 epochs on COCO 2017 for instance segmentation and object detection.

### Detectron2 Instructions
Note: Instructions taken from the MoCo codebase. 

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
