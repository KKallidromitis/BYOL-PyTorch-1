import timm
import torch

model = timm.create_model('swin_tiny_patch4_window7_224',pretrained=True)
torch.save(model.state_dict(),'swin_tiny_patch4_window7_224.pth')