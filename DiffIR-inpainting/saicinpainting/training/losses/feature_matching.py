from typing import List
from torch import nn as nn
import torch
import torch.nn.functional as F


def masked_l2_loss(pred, target, mask, weight_known, weight_missing):
    per_pixel_l2 = F.mse_loss(pred, target, reduction='none')
    pixel_weights = mask * weight_missing + (1 - mask) * weight_known
    return (pixel_weights * per_pixel_l2).mean()


def masked_l1_loss(pred, target, mask, weight_known, weight_missing):
    per_pixel_l1 = F.l1_loss(pred, target, reduction='none')
    pixel_weights = mask * weight_missing + (1 - mask) * weight_known
    return (pixel_weights * per_pixel_l1).mean()


def feature_matching_loss(fake_features: List[torch.Tensor], target_features: List[torch.Tensor], mask=None):
    if mask is None:
        res = torch.stack([F.mse_loss(fake_feat, target_feat)
                           for fake_feat, target_feat in zip(fake_features, target_features)]).mean()
    else:
        res = 0
        norm = 0
        for fake_feat, target_feat in zip(fake_features, target_features):
            cur_mask = F.interpolate(mask, size=fake_feat.shape[-2:], mode='bilinear', align_corners=False)
            error_weights = 1 - cur_mask
            cur_val = ((fake_feat - target_feat).pow(2) * error_weights).mean()
            res = res + cur_val
            norm += 1
        res = res / norm
    return res

def KDLoss(T_fea, S_fea,loss_weight=1.0, temperature = 0.15):
        """
        Args:
            T_fea (List): contain shape (N, L) vector of BlindSR_TA. 
            S_fea (List): contain shape (N, L) vector of BlindSR_ST.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        loss_distill_dis = 0
        loss_distill_abs = 0
        for i in range(len(T_fea)):
            student_distance = F.log_softmax(S_fea[i] / temperature, dim=1)
            teacher_distance = F.softmax(T_fea[i].detach()/ temperature, dim=1)
            loss_distill_dis += F.kl_div(
                        student_distance, teacher_distance, reduction='batchmean')
            loss_distill_abs += nn.L1Loss()(S_fea[i], T_fea[i].detach())
        return loss_weight * loss_distill_dis, loss_weight * loss_distill_abs
