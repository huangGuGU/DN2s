# encoding:utf-8
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, pred, targets):
        num = targets.size(0)
        smooth = 1

        m1 = pred.view(num, -1)
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

        cost = 1 - torch.sum(vpred * vtarget, dim=2) / \
            (torch.sqrt(torch.sum(vpred ** 2, dim=2))
                * torch.sqrt(torch.sum(vtarget ** 2, dim=2)))
        if self.reduce is True:
            return cost.mean()
        return cost

#
# class focal_loss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
#         """
#         focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
#         步骤详细的实现了 focal_loss损失函数.
#         :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
#         :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
#         :param num_classes:     类别数量
#         :param size_average:    损失计算方式,默认取均值
#         """
#
#         super(focal_loss,self).__init__()
#         self.size_average = size_average
#         if isinstance(alpha,list):
#             assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
#             print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
#             self.alpha = torch.Tensor(alpha)
#         else:
#             assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
#             print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
#             self.alpha = torch.zeros(num_classes)
#             self.alpha[0] += alpha
#             self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
#         self.gamma = gamma
#
#     def forward(self, preds, labels):
#         """
#         focal_loss损失计算
#         :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
#         :param labels:  实际类别. size:[B,N] or [B]
#         :return:
#         """
#         # assert preds.dim()==2 and labels.dim()==1
#         preds = preds.view(-1,preds.size(-1))
#         self.alpha = self.alpha.to(preds.device)
#         preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
#         preds_logsoft = torch.log(preds_softmax)
#         preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
#         preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
#         self.alpha = self.alpha.gather(0,labels.view(-1))
#         loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
#         loss = torch.mul(self.alpha, loss.t())
#         if self.size_average:
#             loss = loss.mean()
#         else:
#             loss = loss.sum()
#         return loss

from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



def poly1_cross_entropy(logits, labels, epsilon): # epsilon >=-1.
    # pt, CE, and Poly1 have shape [batch].
    pt = torch.sum(labels * torch.nn.Softmax(logits), dim=-1)
    CE = torch.nn.CrossEntropyLoss(labels, logits)
    Poly1 = CE + epsilon * (1 - pt)
    return Poly1


def train_loss(loss_name, logger):
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
    elif name == 'focal':
        return FocalLoss()
    elif name == 'poly':
        return poly1_cross_entropy()
    else:
        logger.error('The loss function name: {} is invalid !!!!!!'
                     .format(name))
        sys.exit()
