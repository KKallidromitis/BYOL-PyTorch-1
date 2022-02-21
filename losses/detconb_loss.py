import numpy as np
import torch
from torch import nn

class DetconInfoNCECriterion(nn.Module):

    def __init__(self, config):
        super(DetconInfoNCECriterion, self).__init__()
        self.temperature = config['loss']['temperature']
        self.batch_size = config['data']['train_batch_size']
        self.num_rois = config['loss']['mask_rois']
        self.max_val = 1e9
        self.config = config

    def make_same_obj(self,ind_0, ind_1):
        b = ind_0.shape[0]
        same_obj = torch.eq(ind_0.reshape([b, self.num_rois, 1]),
                             ind_1.reshape([b, 1, self.num_rois]))
        return same_obj.float().unsqueeze(2)

    def manual_cross_entropy(self,labels, logits, weight):
        ce = - weight * torch.sum(labels * torch.nn.functional.log_softmax(logits,dim = 1), dim=-1)
        return torch.mean(ce)

    def forward(self, target, pred, tind, pind):        
        #Note: We don't gather from all gpus
        #import ipdb;ipdb.set_trace()
        target1,target2 = target[:self.batch_size],target[self.batch_size:]
        pred1,pred2 = pred[:self.batch_size],pred[self.batch_size:]
        tind1,tind2 = tind[:self.batch_size].to('cuda'),tind[self.batch_size:].to('cuda')
        pind1,pind2 = pind[:self.batch_size].to('cuda'),pind[self.batch_size:].to('cuda')
        
        same_obj_aa = self.make_same_obj(pind1, tind1).to('cuda')
        same_obj_ab = self.make_same_obj(pind1, tind2).to('cuda')
        same_obj_ba = self.make_same_obj(pind2, tind1).to('cuda')
        same_obj_bb = self.make_same_obj(pind2, tind2).to('cuda')
        
        pred1 = torch.nn.functional.normalize(pred1,dim=-1)
        pred2 = torch.nn.functional.normalize(pred2,dim=-1)
        target1 = torch.nn.functional.normalize(target1,dim=-1)
        target2 = torch.nn.functional.normalize(target2,dim=-1)

        labels_local = torch.nn.functional.one_hot(torch.tensor(np.arange(self.batch_size))
                                                   ,self.batch_size).unsqueeze(1).unsqueeze(3).to('cuda')
        labels_ext = torch.nn.functional.one_hot(torch.tensor(np.arange(self.batch_size)),
                                                 self.batch_size * 2).unsqueeze(1).unsqueeze(3).to('cuda')

        logits_aa = torch.einsum("abk,uvk->abuv", pred1, target1) / self.temperature
        logits_bb = torch.einsum("abk,uvk->abuv", pred2, target2) / self.temperature
        logits_ab = torch.einsum("abk,uvk->abuv", pred1, target2) / self.temperature
        logits_ba = torch.einsum("abk,uvk->abuv", pred2, target1) / self.temperature
        
        labels_aa = labels_local * same_obj_aa
        labels_ab = labels_local * same_obj_ab
        labels_ba = labels_local * same_obj_ba
        labels_bb = labels_local * same_obj_bb

        logits_aa = logits_aa - self.max_val * labels_local * same_obj_aa
        logits_bb = logits_bb - self.max_val * labels_local * same_obj_bb
        labels_aa = 0. * labels_aa
        labels_bb = 0. * labels_bb

        labels_abaa = torch.cat([labels_ab, labels_aa], axis=2)
        labels_babb = torch.cat([labels_ba, labels_bb], axis=2)

        labels_0 = torch.reshape(labels_abaa, [self.batch_size, self.num_rois, -1])
        labels_1 = torch.reshape(labels_babb, [self.batch_size, self.num_rois, -1])

        num_positives_0 = torch.sum(labels_0, axis=-1, keepdims=True)
        num_positives_1 = torch.sum(labels_1, axis=-1, keepdims=True)

        labels_0 = labels_0 / torch.max(num_positives_0, torch.ones(num_positives_0.shape).to('cuda'))
        labels_1 = labels_1 / torch.max(num_positives_1, torch.ones(num_positives_1.shape).to('cuda'))

        obj_area_0 = torch.sum(self.make_same_obj(pind1, tind1), axis=[2, 3])
        obj_area_1 = torch.sum(self.make_same_obj(pind2, tind2), axis=[2, 3])

        weights_0 = torch.greater(num_positives_0[..., 0], 1e-3).float()
        weights_0 = weights_0 / obj_area_0
        weights_1 = torch.greater(num_positives_1[..., 0], 1e-3).float()
        weights_1 = weights_1 / obj_area_1

        logits_abaa = torch.cat([logits_ab, logits_aa], axis=2)
        logits_babb = torch.cat([logits_ba, logits_bb], axis=2)

        logits_abaa = torch.reshape(logits_abaa, [self.batch_size, self.num_rois, -1])
        logits_babb = torch.reshape(logits_babb, [self.batch_size, self.num_rois, -1])

        loss_a = self.manual_cross_entropy(labels_0, logits_abaa, weights_0)
        loss_b = self.manual_cross_entropy(labels_1, logits_babb, weights_1)
        loss = loss_a + loss_b

        return loss 