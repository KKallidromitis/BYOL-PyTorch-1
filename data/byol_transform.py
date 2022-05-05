#-*- coding:utf-8 -*-
import torch
from torchvision import transforms
import cv2
from PIL import Image, ImageOps
import numpy as np

class MultiViewDataInjector():
    def __init__(self, transform_list):
        self.transform_list = transform_list
        self.view0_transforms = transforms.Compose([transforms.ToTensor(),     
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])
                                                    ])

    def _random_resized_crop(self, sample, top, left, height, width, size=(224, 224)):
        return transforms.functional.resized_crop(sample, top, left, height, width, size, 
                                                    interpolation=transforms.functional.InterpolationMode.BILINEAR)

    def __call__(self, sample):
        output =  []
        params = []
        for transform in self.transform_list:
            top, left, height, width = transforms.RandomResizedCrop.get_params(sample,scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0))
            sample_i = self._random_resized_crop(sample, top, left, height, width)
            output.append(transform(sample_i).unsqueeze(0))
            params.append((top, left, height, width))
        
        # Get view 0
        top = min([p[0] for p in params])
        bottom = max([p[0] + p[2] for p in params])
        left = min([p[1] for p in params])
        right = max((p[1] + p[3] for p in params))
        h, w = bottom-top, right - left
        assert h*w > 0, f"{top}, {bottom}, {left}, {right}, {params}"
        view0 = self._random_resized_crop(sample, top, left, h, w)
        view0 = self.view0_transforms(view0)
        output.append(view0.unsqueeze(0))

        output_cat = torch.cat(output, dim=0)
        return output_cat

class GaussianBlur():
    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = cv2.GaussianBlur(np.array(img), (self.kernel_size, self.kernel_size), sigma)
        return Image.fromarray(img.astype(np.uint8))

class Solarize():
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, sample):
        return ImageOps.solarize(sample, self.threshold)


def get_transform(stage, gb_prob=1.0, solarize_prob=0.):
    t_list = []
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if stage in ('train', 'val'):
        t_list = [
            # ResizeCrop, Flip handled in MultiViewDataInjector
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(kernel_size=23)], p=gb_prob),
            transforms.RandomApply([Solarize()], p=solarize_prob),
            transforms.ToTensor(),
            normalize]
    else:
        raise NotImplementedError()

    transform = transforms.Compose(t_list)
    return transform
