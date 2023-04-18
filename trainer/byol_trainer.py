#-*- coding:utf-8 -*-
# from math import gamma
from distutils.command.config import config
import os
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import yaml
from torch.utils.data import Subset

from tensorboardX import SummaryWriter
import apex
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

from model import BYOLModel
from optimizer import LARS
from data import ImageLoader,ImageLoadeCOCO
from utils import distributed_utils, params_util, logging_util, eval_util
from utils.data_prefetcher import data_prefetcher
from utils.kmeans.kmeans import KMeans
from utils.scheduler import build_scheduler
from utils.knn import build_imagenet_sampler,kNN

class BYOLTrainer():
    def __init__(self, config):
        self.config = config
        
        """set seed"""
        distributed_utils.set_seed(self.config['seed'])
        
        """device parameters"""
        self.world_size = self.config['world_size']
        self.rank = self.config['rank']
        self.gpu = self.config['local_rank']
        self.distributed = self.config['distributed']

        """get the train parameters!"""
        self.total_epochs = self.config['optimizer']['total_epochs']
        self.warmup_epochs = self.config['optimizer']['warmup_epochs']
        self.overlap_indicator = self.config['data']['overlap_indicator']
        self.use_weight = self.config['data']['weight']
        self.k_means_loss = config['loss']['type'] == 'k-means'
        

        self.train_batch_size = self.config['data']['train_batch_size']
        self.val_batch_size = self.config['data']['val_batch_size']
        self.global_batch_size = self.world_size * self.train_batch_size
        self.legacy_schedule_lr = self.config['optimizer']['legacy_schedule']
        if not self.legacy_schedule_lr:
            self.clustering_scheduler = build_scheduler(self.config['clustering']['scheduler'])
        self.lr_scheduler = build_scheduler(self.config['optimizer']['scheduler'])

        self.num_examples = self.config['data']['num_examples']
        subset = self.config['data'].get("subset", "") #Update num_examples for subsets
        if subset == "imagenet100":
            self.num_examples = 126689
        elif subset == "imagenet1p":
            self.num_examples = 12811
        self.warmup_steps = self.warmup_epochs * self.num_examples // self.global_batch_size
        self.total_steps = self.total_epochs * self.num_examples // self.global_batch_size

        base_lr = self.config['optimizer']['base_lr'] / 256
        self.max_lr = base_lr * self.global_batch_size
        self.lr_type = self.config['optimizer']['lr_type']

        self.base_mm = self.config['model']['base_momentum']
        
        """construct the whole network"""
        self.resume_path = self.config['checkpoint']['resume_path']
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.gpu}')
            torch.cuda.set_device(self.device)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        self.construct_model()

        """save checkpoint path"""
        self.time_stamp = self.config['checkpoint']['time_stamp']
        if self.time_stamp == None:
            self.time_stamp = datetime.datetime.now().strftime('%m_%d_%H-%M')
            
        self.save_epoch = self.config['checkpoint']['save_epoch']
        self.ckpt_path = self.config['checkpoint']['ckpt_path'].format(
            self.time_stamp+'-'+str(np.random.randint(100,999)),self.time_stamp, self.config['model']['backbone']['type'], {})

        save_dir = '/'.join(self.ckpt_path.split('/')[:-1])
        self.log_all = self.config['log']['log_all']
        self.cross_entrophy_loss = torch.nn.CrossEntropyLoss()
        
        """Wandb Log"""
        self.enable_wandb = self.config['log']['enable_wandb']
        if self.enable_wandb:
            import wandb
            if self.gpu==0 or self.log_all:
                wandb.init(project="r2o_pretrain",name = save_dir+'_gpu_'+str(self.rank)+config.get('name',''))
        try:
            os.makedirs(save_dir)
        except:
            pass

        #dump config
        tgt_config_path = os.path.join(save_dir,'config.yaml')
        with open(tgt_config_path,'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

        """log tools in the running phase"""
        self.steps = 0
        self.log_step = self.config['log']['log_step']
        self.ko_leo = self.config['loss']['ko_leo']
        self.logging = logging_util.get_std_logging()
        if self.rank == 0:
            self.writer = SummaryWriter(self.config['log']['log_dir'])

        # eval
        self.knn = config['eval']['knn']
        self.eval_step = config['eval']['eval_step']
        self.num_replicas = config['world_size']
        self.rank = config['rank']
        dataset_eval, _, _ = build_imagenet_sampler(config,self.num_replicas,self.rank)
        n_eval = np.arange(len(dataset_eval))
        knn_train_size = int(0.9 * len(n_eval))
        knn_eval_size = int(0.1 * len(n_eval))
        np.random.shuffle(n_eval)
        idx_eval = n_eval
        idx_eval_train = idx_eval[:knn_train_size]
        idx_eval_test = idx_eval[knn_train_size:knn_train_size+knn_eval_size]
        dataset_eval_train = Subset(dataset_eval,idx_eval_train)
        dataset_eval_test= Subset(dataset_eval,idx_eval_test)
        k_nn_batch_size = 32
        sampler_eval_train = torch.utils.data.DistributedSampler(
            dataset_eval_train, num_replicas=self.num_replicas, rank=self.rank, shuffle=True
        )
        sampler_eval_test = torch.utils.data.DistributedSampler(
            dataset_eval_test, num_replicas=self.num_replicas, rank=self.rank, shuffle=True
        )
        self.data_loader_eval_train = torch.utils.data.DataLoader(dataset_eval_train,batch_size=k_nn_batch_size,sampler=sampler_eval_train,num_workers=4)
        self.data_loader_eval_test = torch.utils.data.DataLoader(dataset_eval_test,batch_size=k_nn_batch_size,sampler=sampler_eval_test,num_workers=4)

    def construct_model(self):
        """get data loader"""
        self.stage = self.config['stage']
        assert self.stage == 'train', ValueError(f'Invalid stage: {self.stage}, only "train" for BYOL training')
        if self.config['data']['mask_type'] == 'coco':
            print("DEBUG: Using Coco GT Mask")
            self.data_ins = ImageLoadeCOCO(self.config)
        else:
            self.data_ins = ImageLoader(self.config)
        self.train_loader = self.data_ins.get_loader(self.stage, self.train_batch_size)

        self.sync_bn = self.config['amp']['sync_bn']
        self.masknet_on = self.config['model']['masknet']
        self.opt_level = self.config['amp']['opt_level']
        print(f"sync_bn: {self.sync_bn}")

        """build model"""
        print("init byol model!")
        net = BYOLModel(self.config)
        if self.sync_bn:
            net = apex.parallel.convert_syncbn_model(net)
        self.model = net.to(self.device)
        print("init byol model end!")

        """build optimizer"""
        print("get optimizer!")
        momentum = self.config['optimizer']['momentum']
        weight_decay = self.config['optimizer']['weight_decay']
        exclude_bias_and_bn = self.config['optimizer']['exclude_bias_and_bn']
        parms = [self.model.online_network, self.model.predictor]
        if self.config['model']['masknet']:
            parms.append(self.model.masknet)
        params = params_util.collect_params(parms,exclude_bias_and_bn=exclude_bias_and_bn)
        self.optimizer = LARS(params, lr=self.max_lr, momentum=momentum, weight_decay=weight_decay)

        """init amp"""
        print("amp init!")
        self.model, self.optimizer = amp.initialize(
            self.model, self.optimizer, opt_level=self.opt_level)

        if self.distributed:
            self.model = DDP(self.model, delay_allreduce=True)
        #cosine_sim = lambda x,y: torch.einsum('nd,md->nm',x,y)
        self.kmeans = KMeans(5)
        self.scale_lr_by_k = self.config['optimizer']['scale_lr_by_k']
        print("amp init end!")

    # resume snapshots from pre-train
    def resume_model(self, model_path=None):
        if model_path is None and not self.resume_path:
            self.start_epoch = 0
            self.logging.info("--> No loaded checkpoint!")
        else:
            model_path = model_path or self.resume_path
            checkpoint = torch.load(model_path, map_location=self.device)

            self.start_epoch = checkpoint['epoch']
            self.steps = checkpoint['steps']
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            amp.load_state_dict(checkpoint['amp'])
            self.logging.info(f"--> Loaded checkpoint '{model_path}' (epoch {self.start_epoch})")

    # save snapshots
    def save_checkpoint(self, epoch):
        if (epoch % self.save_epoch == 0 or epoch == self.total_epochs) and self.rank == 0:
            state = {'config': self.config,
                     'epoch': epoch,
                     'steps': self.steps,
                     'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'amp': amp.state_dict()
                    }
            torch.save(state, self.ckpt_path.format(epoch))
            del state

    def adjust_learning_rate(self, step,k):
        """learning rate warm up and decay"""
        max_lr = self.max_lr
        min_lr = 1e-3 * self.max_lr
        if k > 9000: 
            k = 128

        
        if step < self.warmup_steps:
            lr =  step / int(self.warmup_steps) * max_lr #Following deepmind implementation, returns lr = 0. during first step!

        elif not self.legacy_schedule_lr:
            max_steps = self.total_steps - self.warmup_steps
            global_step = np.minimum((step - self.warmup_steps), max_steps)
            factor = self.lr_scheduler.get_lr( global_step / max_steps * 100)
            lr = max_lr * factor

        elif self.lr_type=='piecewise':
            if step >= (0.96*self.total_steps):
                lr = self.max_lr/10 
            elif step >= (0.98*self.total_steps):
                lr = self.max_lr/100
            else:
                lr = self.max_lr
                
        elif self.lr_type=='cosine': # For lr from detcon paper, returns lr as smalls ~1e-8
            max_steps = self.total_steps - self.warmup_steps
            global_step = np.minimum((step - self.warmup_steps), max_steps)
            cosine_decay_value = 0.5 * (1 + np.cos(np.pi * global_step / max_steps))
            lr = max_lr * cosine_decay_value
        
        if self.scale_lr_by_k:
            lr = lr * k / self.scale_lr_by_k
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_mm(self, step):
        self.mm = 1 - (1 - self.base_mm) * (np.cos(np.pi * step / self.total_steps) + 1) / 2
        
    def forward_loss(self, preds, targets,masks,raw_mask,mask_target,mask_weights):
        if self.k_means_loss:
            return self._forward_k_means_loss(preds, targets,masks,raw_mask,mask_target)
        else:
            return self._forward_masked_byol_loss(preds, targets,masks,raw_mask,mask_target,mask_weights)
    
    def _forward_masked_byol_loss(self, preds, targets,masks,raw_mask,mask_target,mask_weights):
        zero = torch.tensor(0.0)
        weights = masks.sum(dim=-1).detach()
        mask_batch_size = masks.shape[0] // 2
        mask_exists = torch.logical_and(weights[:mask_batch_size]>1e-3,weights[mask_batch_size:]>1e-3).float()
        if self.use_weight:
            weights =  (weights[:mask_batch_size]+weights[mask_batch_size:])/2
        else:
            weights = torch.ones_like(weights[:mask_batch_size])
        if self.overlap_indicator:
            weights *= mask_exists
        weights = weights.repeat([2,1])
        preds = F.normalize(preds, dim=-1) 
        targets = F.normalize(targets, dim=-1) 
        mask_weights = mask_weights.repeat([2,1])
        #weights =  (weights * (mask_weights>0)).sum()
        inv_loss = ((preds-targets)**2).sum(dim=-1) * weights * mask_weights
        #Masked out area
        if self.ko_leo > 0:
            diag_msk = torch.eye(preds.shape[1]).to(preds.device).unsqueeze(0) # * (1-torch.einsum('ba,bc->bac',weights,weights))
            dist_mat = torch.cdist(preds,targets) * (1-diag_msk) + diag_msk * 10 # pariwise dist with diag larger than 0 
            avg_dist = dist_mat.min(-1).values.log().mean()
        else:
            avg_dist = 2.0

        #breakpoint()
        if weights.sum() == 0:
            inv_loss = torch.FloatTensor(0.0,requires_grad=True).cuda()
        else:
            inv_loss = inv_loss.sum() / weights.sum()     
        total_loss = inv_loss   
        if self.ko_leo > 0:
            total_loss += ( 2 - self.ko_leo * avg_dist)
        return  total_loss,avg_dist,zero,inv_loss,torch.tensor(0.0),mask_exists.float().sum(-1).mean().detach()
    
    def _forward_k_means_loss(self, preds, targets,masks,raw_mask,mask_target):
        zero = torch.tensor(0.0)
        weights = masks.sum(dim=-1).detach()
        mask_batch_size = masks.shape[0] // 2
        mask_exists = torch.logical_and(weights[:mask_batch_size]>1e-3,weights[mask_batch_size:]>1e-3).float()
        if self.use_weight:
            weights =  (weights[:mask_batch_size]+weights[mask_batch_size:])/2
        else:
            weights = torch.ones_like(weights[:mask_batch_size])
        if self.overlap_indicator:
            weights *= mask_exists
        weights = weights.repeat([2,1])
        preds = F.normalize(preds, dim=-1) 
        targets = F.normalize(targets, dim=-1) 
        b,n_p,h = preds.shape # B X N_pixelx X Dim
        _,n_k,_ = targets.shape
        inv_loss = ((preds.view(b,n_p,1,h)-targets.view(b,1,n_k,h))**2).sum(dim=-1)  # B X N_pixelx X n_k 
        averaged_mask = masks / masks.sum(dim=1,keepdims=True) # B X N_k X N_pixelx
        averaged_mask = averaged_mask.permute(0,2,1)# B X  N_pixelx X N_k 
        inv_loss = (inv_loss * averaged_mask).sum(dim=1) # B X  X N_k 
        inv_loss *= weights
        if weights.sum() == 0:
            inv_loss = torch.FloatTensor(0.0,requires_grad=True).cuda()
        else:
            inv_loss = inv_loss.sum() / weights.sum()        
    
        return  inv_loss,zero,zero,inv_loss,torch.tensor(0.0),mask_exists.float().sum(-1).mean().detach()
    
    
    def run_knn(self,force=False,epoch=0):
        if self.knn > 0 or force:
            self.model.eval()
            net = self.model.module.online_network.encoder
            net.eval()
            kNN(net,self.data_loader_eval_train,self.data_loader_eval_test,self.knn,epoch=epoch)
            net.train()
            del net
            self.model.train()

    def train_epoch(self, epoch, printer=print):
        batch_time = eval_util.AverageMeter()
        data_time = eval_util.AverageMeter()
        forward_time = eval_util.AverageMeter()
        backward_time = eval_util.AverageMeter()
        log_time = eval_util.AverageMeter()
        loss_meter = eval_util.AverageMeter()

        self.model.train()

        end = time.time()
        self.data_ins.set_epoch(epoch)
        prefetcher = data_prefetcher(self.train_loader)
        images, masks,diff_transfrom = prefetcher.next()
        i = 0
        use_masknet = False # epoch > 30
        #breakpoint()
        if self.steps % self.eval_step == 0 and self.knn > 0:
                self.model.eval()
                net = self.model.module.online_network.encoder
                net.eval()
                kNN(net,self.data_loader_eval_train,self.data_loader_eval_test,self.knn,epoch=epoch)
                net.train()
                del net
                self.model.train()
        while images is not None:
            i += 1
            clustering_k = self.clustering_scheduler.get_num_segments(epoch)
            self.adjust_learning_rate(self.steps,clustering_k)
            self.adjust_mm(self.steps)
            self.steps += 1
            #import ipdb;ipdb.set_trace()
            assert images.dim() == 5, f"Input must have 5 dims, got: {images.dim()}"
            view1 = images[:, 0, ...].contiguous()
            view2 = images[:, 1, ...].contiguous()
            view_raw = images[:, 2, ...].contiguous()
            # mask B X (3 Views) X (2 channels [intersection, SLIC]  ) X H X W
            input_masks = masks[:,:2,0,...].contiguous() # discard last mask,B X 2 X 224 X 224
            slic_labelmap = masks[:,2,1,...].contiguous() # B X 1 X H X W
            full_view_prior_mask = masks[:,2,0,...].contiguous() # B X 1 X H X W
            # measure data loading time
            data_time.update(time.time() - end)
            #breakpoint()
            # forward
            tflag = time.time()
            #breakpoint()
            q, target_z,pinds, tinds,down_sampled_masks,raw_mask,mask_target,num_segs,applied_mask,mask_weights = self.model(view1, view2, self.mm, input_masks,view_raw,diff_transfrom,slic_labelmap,use_masknet,full_view_prior_mask,
            clustering_k=clustering_k)
            forward_time.update(time.time() - tflag)

            tflag = time.time()
            loss,eh_obj,eh_dist,inv_loss,mask_loss,num_indicator = self.forward_loss(q,target_z,down_sampled_masks,raw_mask,mask_target,mask_weights)

            self.optimizer.zero_grad()
            if self.opt_level == 'O0':
                loss.backward()
            else:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            self.optimizer.step()
            backward_time.update(time.time() - tflag)
            loss_meter.update(loss.item(), view1.size(0))

            tflag = time.time()
            if self.steps % self.log_step == 0 and self.rank == 0:
                self.writer.add_scalar('lr', round(self.optimizer.param_groups[0]['lr'], 5), self.steps)
                self.writer.add_scalar('mm', round(self.mm, 5), self.steps)
                self.writer.add_scalar('loss', loss_meter.val, self.steps)
            log_time.update(time.time() - tflag)

            batch_time.update(time.time() - end)
            end = time.time()
            #import ipdb;ipdb.set_trace()
            # Print log info
            # Print log info
            if (self.gpu == 0 or self.log_all) and self.steps % self.log_step == 0:
                if self.enable_wandb:
                    import wandb
                    from utils.visualize import wandb_dump_img
                    # Log per batch stats to wandb (average per epoch is also logged at the end of function)
                    wandb.log({
                        'lr': round(self.optimizer.param_groups[0]["lr"], 5),
                        'mm': round(self.mm, 5),
                        'loss': round(loss_meter.val, 5),
                        "eh_obj":round(eh_obj.item(),5),
                        "eh_dist":round(eh_dist.item(),5),
                        "inv_loss":round(inv_loss.item(),5),
                        "mask_loss":round(mask_loss.item(),5),
                        "num_segs":round(num_segs.item(),5),
                        'Batch Time': round(batch_time.val, 5),
                        'Data Time': round(data_time.val, 5),
                        "K-clustering":clustering_k,
                        "num_indicator":round(num_indicator.item(),5),
                        'Forward Time': round(forward_time.val, 5),
                        'Backward Time': round(backward_time.val, 5),
                    })
                if  (self.steps//self.log_step) % 5 == 1:
                    # img_mask = mask_target[0].detach().cpu()
                    # applied_mask = applied_mask[0].detach().cpu()

                    # view_raw = np.exp(view_raw[0].permute(1,2,0).detach().cpu())
                    # wandb_dump_img([view_raw,img_mask,applied_mask],"Masks")


                    img_mask = mask_target[0].detach().cpu()
                    applied_mask = applied_mask[0].detach().cpu()

                    view_raw = np.exp(view_raw[0].permute(1,2,0).detach().cpu())
                    mask_visual = raw_mask[0].permute(1,2,0) 
                    mh,mw,mc = mask_visual.shape
                    # mask_visual = mask_visual.view(mh*mw,mc)
                    # mask_visual = self.kmeans.fit_transform(mask_visual).view(mh,mw).detach().cpu()
                    if self.enable_wandb:
                        wandb_dump_img([view_raw,img_mask,applied_mask],"Masks")

                printer(f'Epoch: [{epoch}][{i}/{len(self.train_loader)}]\t'
                        f'Step {self.steps}\t'
                        f'lr {round(self.optimizer.param_groups[0]["lr"], 5)}\t'
                        f'mm {round(self.mm, 5)}\t'
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'Batch Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'Data Time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                        f'Forward Time {forward_time.val:.4f} ({forward_time.avg:.4f})\t'
                        f'Backward Time {backward_time.val:.4f} ({backward_time.avg:.4f})\t'
                        f'Log Time {log_time.val:.4f} ({log_time.avg:.4f})\t')

            images, masks,diff_transfrom = prefetcher.next()
        if self.enable_wandb:
            if self.gpu == 0 or self.log_all: 
                # Log averages at end of Epoch
                wandb.log({
                    'Average Loss (Per-Epoch)': round(loss_meter.avg, 5),
                    'Average Batch-Time (Per-Epoch)': round(batch_time.avg, 5),
                    'Average Data-Time (Per-Epoch)': round(data_time.avg, 5),
                    'Average Forward-Time (Per-Epoch)': round(forward_time.avg, 5),
                    'Average Backward-Time (Per Epoch)': round(backward_time.avg, 5),
                })

