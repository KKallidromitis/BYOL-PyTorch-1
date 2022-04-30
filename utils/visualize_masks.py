import wandb
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def wandb_set(img1,img2,category):
    img1,img2 = img1.detach().cpu().numpy(),img2.detach().cpu().numpy()
    
    fig, axes = plt.subplots(1,2,figsize=(10, 5))
    fig.tight_layout()
    axes[0].imshow(img1)
    axes[1].imshow(img2)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.close()
    wandb.log({category:wandb.Image(fig)}) 
    return

def wandb_sample(mask_rois,pool_size,img1,img2,category):
    img1 = torch.reshape(img1,(mask_rois,pool_size,pool_size))
    img2 = torch.reshape(img2,(mask_rois,pool_size,pool_size))
    img1,img2 = img1.detach().cpu().numpy(),img2.detach().cpu().numpy()
    n = img1.shape[0]
    
    fig, axes = plt.subplots(2,n,figsize=(20,5))
    fig.tight_layout()
    for i in range(n):
        axes[0][i].imshow(img1[i])
        axes[1][i].imshow(img2[i])
        axes[0][i].set_xticks([]);axes[0][i].set_yticks([])
        axes[1][i].set_xticks([]);axes[1][i].set_yticks([])
    plt.close()
    wandb.log({category:wandb.Image(fig)}) 
    return