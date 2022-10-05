from torchvision import datasets
import os
import torch
from torch.utils.data._utils.collate import default_collate
from .misc import is_main_process
import torchvision.transforms as transforms

class ImageNet100Dataset(datasets.ImageFolder):

    def __init__(self,path,anno_file,transform) -> None:
        super(ImageNet100Dataset,self).__init__(path,transform=transform)
        self.imgs = self.samples
        with open(anno_file) as f:
            files_100 = f.readlines()
        #breakpoint()
        files_100 = [x.replace('\n','') for x in files_100]
        new_samples = []
        for x,y in self.samples:
            if any([t in x for t in files_100]):
                new_samples.append((x,y))
        self.samples = new_samples


def build_imagenet_sampler(config, num_replicas, rank):
    eval_transforms = transforms.Compose([
            transforms.RandomResizedCrop(config['data']['resize_size'], scale=(1.0, 1.0), interpolation=3),  # 3 is bicubic
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_dir = config['eval']['img_dir']
    dataset = ImageNet100Dataset(os.path.join(img_dir,'train'),anno_file='imagenet100.txt',transform=eval_transforms)
    sampler =  torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=True
    )
    collate_fn = default_collate
    return dataset, sampler, collate_fn

import imp
import torch
import time
import wandb

from torch.nn.functional import adaptive_avg_pool2d
from tqdm.cli import tqdm
import torch.distributed
import os

from .distributed_utils import gather_from_all
from torch.distributed import all_reduce
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=1)
    return output

def get_knn_iter(x,gpu):
    if gpu == 0:
        return tqdm(x)
    else:
        return x

@torch.no_grad()
def kNN(net, trainloader, testloader, K, sigma=0.07, feat_dim=2048, gpu=None):
    net.eval()
    gpu = int(os.environ['LOCAL_RANK'])
    print(f"Starting KNN evaluation with K={K}")



    st_time = time.time()
    trainFeatures = torch.zeros(
        [feat_dim + 1, len(trainloader)*trainloader.batch_size])
    if gpu is not None:
        trainFeatures = trainFeatures.cuda(gpu)
    else:
        trainFeatures = trainFeatures.cuda()

    for batch_idx, (inputs, targets) in get_knn_iter(enumerate(trainloader),gpu):
        # targets = targets.cuda(async=True)
        batchSize = inputs.size(0)
        if gpu is not None:
            inputs = inputs.cuda(gpu)
        features = net(inputs)
        features = features.mean((2,3))
        trainFeatures[:-1, batch_idx*batchSize:batch_idx *
                      batchSize+batchSize] = features.T
        trainFeatures[-1, batch_idx*batchSize:batch_idx *
                      batchSize+batchSize] = targets

    # TODO(cjrd) let's clean this up - a bit cuda-messy
    print(f"distributed world size: {torch.distributed.get_world_size()}")
    if gpu is None:
        # TODO(cjrd) this seems broken
        #trainFeatures = concat_all_gather(trainFeatures)
        trainFeatures = gather_from_all(trainFeatures.permute(1,0).contiguous()).permute(1,0)
        trainLabels = torch.flatten(trainFeatures[-1, :]).cuda()
        trainFeatures = trainFeatures[:-1, :].cuda()
    else:
        trainFeatures = gather_from_all(trainFeatures.permute(1,0).contiguous()).permute(1,0)
        trainLabels = torch.flatten(trainFeatures[-1, :]).cuda(gpu)
        trainFeatures = trainFeatures[:-1, :].cuda(gpu)

    trainFeatures = torch.nn.functional.normalize(trainFeatures,dim=0)
    
    print(
        f"Grabbing all kNN training features took {(time.time() - st_time): .1f} seconds")
    print(f"Shape of final train features {trainFeatures.shape}")
    top1 = torch.FloatTensor([0.,])
    total = torch.FloatTensor([0.,])
    if gpu is not None:
        top1 = top1.cuda(gpu)
        total = total.cuda(gpu)
    else:
        top1 = top1.cuda()
        total = total.cuda()
    C = int(trainLabels.max() + 1)
    st_time = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets) in get_knn_iter(enumerate(testloader),gpu):

            # targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            if gpu is not None:
                inputs = inputs.cuda(gpu)
                targets = targets.cuda(gpu)
            features = net(inputs)
            features = features.mean((2,3))
            features = torch.nn.functional.normalize(features, dim=1)
            dist = torch.mm(features, trainFeatures)
            # if misc.is_main_process():
            #     breakpoint()
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi).long()

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(
                batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)
            
            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()

            total += targets.size(0)
    all_reduce(top1)
    all_reduce(total)
    top1 = top1.detach().cpu().numpy().item() # sum
    total = total.detach().cpu().numpy().item() #sum
    if is_main_process():
        print(
            f"Evaluating all kNN took an additional {(time.time() - st_time):.1f} seconds")
        print("knn results")
        print(top1*100./total,top1,total)
        wandb.log(
           { "Knn-ACC":top1*100./total}
        )

    return top1/total