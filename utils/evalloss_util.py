# encoding:utf-8
import sys
import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = logits
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class NpccLoss(nn.Module):
    def __init__(self, reduction=True):
        super(NpccLoss, self).__init__()
        self.reduce = reduction

    def forward(self, pred, target):
        target = target.view(target.size(0), target.size(1), -1)
        pred = pred.view(pred.size(0), pred.size(1), -1)

        vpred = pred - torch.mean(pred, dim=2).unsqueeze(-1)
        vtarget = target - torch.mean(target, dim=2).unsqueeze(-1)

        cost = - torch.sum(vpred * vtarget, dim=2) / \
            (torch.sqrt(torch.sum(vpred ** 2, dim=2))
                * torch.sqrt(torch.sum(vtarget ** 2, dim=2)))
        if self.reduce is True:
            return cost.mean()
        return cost


def eval_loss(loss_name, logger):
    name = loss_name
    if name == 'mse':
        return nn.MSELoss()
    elif name == 'mae':
        return nn.L1Loss()
    elif name == 'ce':
        # return nn.BCELoss()
        return nn.CrossEntropyLoss()
    elif name == 'dice':
        return SoftDiceLoss()
    elif name == 'npcc':
        return NpccLoss()
    else:
        logger.error('The eval loss name: {} is invalid !!!!!!'
                     .format(name))
        sys.exit()
