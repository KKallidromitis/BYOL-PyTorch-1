"""
File from detectron2. Used to convert ResNet checkpoint -> Detectron2 compatible ResNet checkpoint
Adapted for the checkpoints from this codebase
"""
#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import pickle as pkl
import sys
import torch

"""
Usage:
  # download one of the ResNet{18,34,50,101,152} models from torchvision:
  wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O r50.pth
  # run the conversion
  ./convert-torchvision-to-d2.py r50.pth r50.pkl

  # Then, use r50.pkl with the following changes in config:

MODEL:
  WEIGHTS: "/path/to/r50.pkl"
  PIXEL_MEAN: [123.675, 116.280, 103.530]
  PIXEL_STD: [58.395, 57.120, 57.375]
  RESNETS:
    DEPTH: 50
    STRIDE_IN_1X1: False
INPUT:
  FORMAT: "RGB"

  These models typically produce slightly worse results than the
  pre-trained ResNets we use in official configs, which are the
  original ResNet models released by MSRA.
"""

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")["model"]

    newmodel = {}
    for k in list(obj.keys()):
        if "module.online_network.encoder." not in k: #Only use online network encoder param
            obj.pop(k)
            continue
        else:
            old_k = k
            k = k[len("module.online_network.encoder."):] #Remove byol model prefix
        
        if k[0] in ["0", "1"]:
            if k[0] == "0": #To match typical torchvision convention (our 0 -> conv1)
                k = "conv1" + k[1:]
            else: # bn1 param
                k = "bn1" + k[1:]
            k = "stem." + k
            
        for t in [1, 2, 3, 4]:
            k = k[0].replace("{}".format(t+3), "res{}".format(t + 1)) + k[1:]
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())
    else:
        print("All keys found and converted!")