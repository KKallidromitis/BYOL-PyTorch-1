from PIL import Image
import cv2
import os
from torch.utils.data import Dataset
import yaml
import numpy as np
from torchvision.transforms import ToTensor,Compose,Normalize,Resize,CenterCrop
#from torchvision.transforms import InterpolationMode
import PIL.Image as InterpolationMode


class CUBDataset(Dataset):

    def __init__(self,image_root,anno_root,split_file='/shared/jacklishufan/CUB/train_val_test_split.txt',
    image_file='/shared/jacklishufan/CUB/CUB_200_2011/images.txt',
    split='test'
    ) -> None:
        super().__init__()
        self.transform = Compose([
                        # Resize(320,interpolation=InterpolationMode.BILINEAR),
                        # CenterCrop(320),
            ToTensor(),
                        Normalize(
                mean=[123.675/255, 116.28/255, 103.53/255],
                std=[58.395/255, 57.12/255, 57.375/255],
            ),

            
        ])
        self.transform_label = Compose([
            #  Resize(320,interpolation=InterpolationMode.NEAREST),
            #  CenterCrop(320),
            ToTensor(),
           
            
        ])
        files_dirs = list(os.walk(anno_root))
        all_files = []
        for root,dirs,files in files_dirs:
            for file in files:
                if '.jpg' in file.lower() or '.png' in file.lower():
                    all_files.append(os.path.join(root,file).replace(anno_root+'/',''))
#             print(file)
        self.image_root = image_root
        self.anno_root = anno_root
        self.files = all_files
        if split=='test':
            with open(split_file) as f:
                splits = f.readlines()
            test = []
            with open('/shared/jacklishufan/CUB/CUB_200_2011/images.txt') as f:
                all_files = f.readlines()
            all_files = list(os.path.join(x.replace('\n','').split(' ')[1]) for x in all_files if x)
            for r,f in zip(splits,all_files):
                x,y = r.replace('\n','').split(' ')
                if int(y) ==2:
                    test.append(f)
            self.files = test
            #print(test)

        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_root,self.files[index])
        anno_path = os.path.join(self.anno_root,self.files[index].replace('jpg','png'))
        image = Image.open(img_path).convert('RGB')
        label = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
        return self.transform(image),self.transform_label(label)
    
class FlowerDataset(Dataset):

    def __init__(self,image_root,anno_root) -> None:
        super().__init__()
        self.transform = Compose([
                        # Resize(320,interpolation=InterpolationMode.BILINEAR),
                        # CenterCrop(320),
            ToTensor(),
                        Normalize(
                mean=[123.675/255, 116.28/255, 103.53/255],
                std=[58.395/255, 57.12/255, 57.375/255],
            ),

            
        ])
        self.transform_label = Compose([
            #  Resize(320,interpolation=InterpolationMode.NEAREST),
            #  CenterCrop(320),
            ToTensor(),
           
            
        ])
        files_dirs = list(os.walk(anno_root))
        all_files = []
        for root,dirs,files in files_dirs:
            for file in files:
                if '.jpg' in file.lower() or '.png' in file.lower():
                    all_files.append(os.path.join(root,file).replace(anno_root+'/',''))
        self.image_root = image_root
        self.anno_root = anno_root
        self.files = all_files

        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_root,self.files[index].replace('segmim','image'))
        anno_path = os.path.join(self.anno_root,self.files[index])
        image = Image.open(img_path).convert('RGB')
        label = (cv2.imread(anno_path) - np.array([254,0,0]).reshape(1,1,-1))
        label = label.max(-1)
        label = label<0.2*255
        label = (~label).astype(np.uint8)*255
        return self.transform(image),self.transform_label(label)