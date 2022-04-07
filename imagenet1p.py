#Generate 1percent Imagenet Dataset
#use wget https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/1percent.txt

import os
import PIL
import torchvision
from tqdm import tqdm

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    
    https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def main(imagenet_dir,output_dir,txt_path):    
    dataset = ImageFolderWithPaths(imagenet_dir)

    with open(txt_path) as f:
        image_ids = f.read().splitlines() 

    for obj in tqdm(dataset):
        img = obj[0]
        name = obj[2].split('/')[-1]

        if name in image_ids:
            save_path = os.path.join(output_dir,name.split('_')[0])
            try:
                img.save(os.path.join(save_path,name))
            except:
                os.makedirs(save_path)
                img.save(os.path.join(save_path,name))
    return
            
if __name__=="__main__":
    
    #Change Location here
    imagenet_dir = '/home/kkallidromitis/data/imagenet/images/train'
    output_dir = '/home/kkallidromitis/data/sample/images/train'
    txt_path = '/home/kkallidromitis/data/imagenet/raw/1percent.txt'
    
    main(imagenet_dir,output_dir,txt_path)