import torch
import sys
import yaml
from torchvision import transforms, datasets
import torchvision
import numpy as np
import os
from sklearn import preprocessing
from torch.utils.data.dataloader import DataLoader
import tqdm
with tqdm.cli.tqdm(total=100) as pbar:     
     for i in range(10):
         pbar.update(10)
         pbar.refresh()

import torch
from torchvision import models

class ResNet(torch.nn.Module):
    def __init__(self, net_name, pretrained=False, use_fc=False):
        super().__init__()
        base_model = models.__dict__[net_name](pretrained=pretrained)
        self.encoder = torch.nn.Sequential(*list(base_model.children())[:-1])

        self.use_fc = use_fc
        if self.use_fc:
            self.fc = torch.nn.Linear(2048, 512)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        if self.use_fc:
            x = self.fc(x)
        return x
    
device = torch.device('cuda:0')
model = ResNet('resnet50', pretrained=False, use_fc=False).to(device)
model_path = '/home/jacklishufan/detconb/ckpt/detconb/04_24_17-51/04_24_17-51_resnet50_300.pth.tar'
pth_file = torch.load(model_path, map_location=device)
checkpoint = torch.load(model_path, map_location=device)['model']#['online_backbone']
state_dict = {}
length = len(model.encoder.state_dict())
for name, param in zip(model.encoder.state_dict(), list(checkpoint.values())[:length]):
    state_dict[name] = param
model.encoder.load_state_dict(state_dict, strict=True)
#model =  torch.nn.DataParallel(model, device_ids=[1, 2, 3,4,5,6,7,8])
model.eval()
batch_size = 512 
data_transforms = torchvision.transforms.Compose([
                                                    transforms.Resize((224, 224)), #FIXME: They only did smallest side resize to 224
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                    
                                                ])
train_dataset = datasets.ImageFolder('../imagenet/images/train/', transform=data_transforms)

test_dataset = datasets.ImageFolder('../imagenet/images/val/', transform=data_transforms)

print("Input shape:", train_dataset.__getitem__(0)[0].shape)
print("Input shape:", test_dataset.__getitem__(0)[0].shape)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          num_workers=8, drop_last=False, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size,
                          num_workers=8, drop_last=False, shuffle=True)

# device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
# encoder = ResNet18(**config['network'])
encoder = model
output_feature_dim = 2048 #encoder.projetion.net[0].in_features


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

logreg = LogisticRegression(output_feature_dim, 1000)
logreg = logreg.to(device)


def get_features_from_encoder(encoder, loader):
    
    x_train = []
    y_train = []

    # get the features from the pre-trained model
    for (x, y) in tqdm.cli.tqdm((loader)):
        with torch.no_grad():
            feature_vector = encoder(x.to(device))
            x_train.extend(feature_vector.cpu())
            y_train.extend(y.cpu().numpy())

    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)
    return x_train, y_train

encoder.eval()
print("Getting Train Features")
x_train, y_train = get_features_from_encoder(encoder, train_loader)
# print("Getting Test Features")
# x_test, y_test = get_features_from_encoder(encoder, test_loader)
print("Getting Test Features")
x_test, y_test = get_features_from_encoder(encoder, test_loader)
if len(x_train.shape) > 2:
    x_train = torch.mean(x_train, dim=[2, 3])
    x_test = torch.mean(x_test, dim=[2, 3])
    
print("Training data shape:", x_train.shape, y_train.shape)
print("Testing data shape:", x_test.shape, y_test.shape)
if len(x_train.shape) > 2:
    x_train = torch.mean(x_train, dim=[2, 3])
    x_test = torch.mean(x_test, dim=[2, 3])
    
print("Training data shape:", x_train.shape, y_train.shape)
print("Testing data shape:", x_test.shape, y_test.shape)
def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test):

    train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
    return train_loader, test_loader

train_loader, test_loader = create_data_loaders_from_arrays(x_train, y_train, x_test, y_test)
#train_loader, test_loader = create_data_loaders_from_arrays(torch.from_numpy(x_train), y_train, torch.from_numpy(x_test), y_test)

optimizer = torch.optim.Adam(logreg.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()
eval_every_n_epochs = 10

for epoch in tqdm.cli.tqdm((range(100))):
#     train_acc = []
    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()        
        
        logits = logreg(x)
        predictions = torch.argmax(logits, dim=1)
        
        loss = criterion(logits, y)
        
        loss.backward()
        optimizer.step()
    
    total = 0
    if epoch % eval_every_n_epochs == 0 or epoch==99:
        correct = 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = logreg(x)
            predictions = torch.argmax(logits, dim=1)
            
            total += y.size(0)
            correct += (predictions == y).sum().item()
            
        acc = 100 * correct / total
        print(f"Testing accuracy: {np.mean(acc)}")