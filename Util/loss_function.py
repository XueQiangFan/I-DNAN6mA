#!/Users/11834/.conda/envs/Pytorch_GPU/python.exe
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File    ：FWorks -> lossfunction
@IDE    ：PyCharm
@Date   ：2020/8/21 18:47
=================================================='''
# 仅限回归问题
import torch
# L2 损失
def MSELoss(predict, true):
    return torch.sum((true - predict)**2)
# L1 损失
def MAELoss(predict, true):
    return torch.mean(torch.abs(true - predict))

# huber loss
def HuberLoss(predict, true, delta):
    loss = torch.where(torch.abs(true-predict) <= delta, 0.5*((true-predict)**2), delta*torch.abs(true - predict) - 0.5*(delta**2))
    return torch.mean(loss)
# log cosh loss
def LogcoshLoss(predict, true):
    loss = torch.log(torch.cosh(predict - true))
    return torch.mean(loss)

# 仅限分类问题