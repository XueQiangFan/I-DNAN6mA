#!/Users/11834/.conda/envs/Pytorch_GPU/python.exe
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File    ：RNAsolvent+ -> calculate_16xLxL_matrix_map
@IDE    ：PyCharm
@Author  : Xue-Qiang Fan
@Date   ：2022/1/27 22:37
=================================================='''
import torch
import numpy as np

def constraint_matrix(sequence):
    BASES = 'ACGT'
    bases = np.array([base for base in BASES])
    one_hot_encoding = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[0] * len(BASES)]) for base
         in sequence])
    one_hot_encoding = torch.from_numpy(one_hot_encoding).float()

    base_a = one_hot_encoding[:, 0]
    base_u = one_hot_encoding[:, 1]
    base_c = one_hot_encoding[:, 2]
    base_g = one_hot_encoding[:, 3]
    length = base_a.shape[0]

    aa = torch.matmul(base_a.view(length, 1), base_a.view(1, length))
    au = torch.matmul(base_a.view(length, 1), base_u.view(1, length))
    ac = torch.matmul(base_a.view(length, 1), base_c.view(1, length))
    ag = torch.matmul(base_a.view(length, 1), base_g.view(1, length))

    ua = torch.matmul(base_u.view(length, 1), base_a.view(1, length))
    uu = torch.matmul(base_u.view(length, 1), base_u.view(1, length))
    uc = torch.matmul(base_u.view(length, 1), base_c.view(1, length))
    ug = torch.matmul(base_u.view(length, 1), base_g.view(1, length))

    ca = torch.matmul(base_c.view(length, 1), base_a.view(1, length))
    cu = torch.matmul(base_c.view(length, 1), base_u.view(1, length))
    cc = torch.matmul(base_c.view(length, 1), base_c.view(1, length))
    cg = torch.matmul(base_c.view(length, 1), base_g.view(1, length))

    ga = torch.matmul(base_g.view(length, 1), base_a.view(1, length))
    gu = torch.matmul(base_g.view(length, 1), base_u.view(1, length))
    gc = torch.matmul(base_g.view(length, 1), base_c.view(1, length))
    gg = torch.matmul(base_g.view(length, 1), base_g.view(1, length))

    one_hot_encoding = one_hot_encoding.numpy()
    map = torch.stack([aa, au, ac, ag, ua, uu, uc, ug, ca, cu, cc, cg, ga, gu, gc, gg], dim=0).numpy()

    return map
