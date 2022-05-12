#!/Users/11834/.conda/envs/Pytorch_GPU/python.exe
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File    ：FWorks -> CalEvalutionIndex
@IDE    ：PyCharm
@Date   ：2020/8/17 18:28
=================================================='''
import math
import torch
import numpy as np
from numba import jit


@jit
def calculate_confusion_matrix(TP, FP, TN, FN):
    # Sen = TP / (TP + FN)  # 真正率 查全率 Racall
    # Spe = TN / (TN + FP)  # 真负率
    # Pre = TP / (TP + FP)  # 精确率 查准率
    # Acc = (TP + TN) / (TP + FN + TN + FP)  # 准确率
    # Mcc = (TP * TN - FN * FP) / math.sqrt((TP + FN) * (TP + FP) * (TN + FN) * (TN + FP))
    # FOne = (2 * Pre * Sen) / (Pre + Sen)
    if (TP + FN) == 0:
        Sen = 0
    else:
        Sen = TP / (TP + FN)  # 真正率 查全率 Racall

    if (TN + FP) == 0:
        Spe = 0
    else:
        Spe = TN / (TN + FP)  # 真负率

    if (TP + FP) == 0:
        Pre = 0
    else:
        Pre = TP / (TP + FP)  # 精确率 查准率

    if (TP + FN + TN + FP) == 0:
        Acc = 0
    else:
        Acc = (TP + TN) / (TP + FN + TN + FP)  # 准确率

    if (TP + FN) * (TP + FP) * (TN + FN) * (TN + FP) == 0:
        Mcc = 0
    else:
        Mcc = float(TP * TN - FN * FP) / float(math.sqrt((TP + FN) * (TP + FP) * (TN + FN) * (TN + FP)))

    if Pre + Sen == 0:
        FOne = 0
    else:
        FOne = (2 * Pre * Sen) / (Pre + Sen)
        
    return Sen, Spe, Pre, Acc, Mcc, FOne


def calculate_TP_FP_TN_FN(predict, label, threshold=0.5):
    TP, FP, TN, FN = 0, 0, 0, 0
    total_TP, total_TN = 0, 0
    for i in range(label.shape[0]):
        pre, lab = predict[i].item(), label[i].item()
        if pre < threshold:
            pre = 0
        elif threshold <= pre:
            pre = 1

        if lab == 0:
            total_TN += 1
            if pre == 0:
                TN += 1
            elif pre == 1:
                FP += 1
        elif lab == 1:
            total_TP += 1
            if pre == 0:
                FN += 1
            elif pre == 1:
                TP += 1
    return TP, FP, TN, FN, total_TP, total_TN
