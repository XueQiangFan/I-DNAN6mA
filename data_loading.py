#!/Users/11834/.conda/envs/Pytorch_GPU/python.exe
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File    ：master_thesis -> data_loading_
@IDE    ：PyCharm
@Author  : Xue-Qiang Fan
@Date   ：2022/3/7 15:15
=================================================='''
import numpy as np
import random
import xlrd
from I_DNAN6mA_D.calculate_16xLxL_matrix_map import constraint_matrix
from torch.utils.data import DataLoader


def read_seq_label(filename):
    workbook = xlrd.open_workbook(filename=filename)

    booksheet_train = workbook.sheet_by_index(0)
    nrows_train = booksheet_train.nrows

    seq = []
    label = []
    for i in range(nrows_train):
        seq.append(booksheet_train.row_values(i)[0])
        label.append(booksheet_train.row_values(i)[1])

    return seq, np.array(label).astype(int)


def seq_to_one_hot_map(filename):
    seq, label = read_seq_label(filename)

    nrows = len(seq)
    seq_len = len(seq[0])

    seq_one_hot = np.zeros((nrows, seq_len, 4), dtype='int')
    seq_0123 = np.zeros((nrows, seq_len), dtype='int')
    map_16xLxL = np.zeros((nrows, 16, seq_len, seq_len), dtype='int')
    for i in range(nrows):
        cur_seq = seq[i]
        cur_seq = cur_seq.replace('A', '0')
        cur_seq = cur_seq.replace('C', '1')
        cur_seq = cur_seq.replace('G', '2')
        cur_seq = cur_seq.replace('T', '3')
        seq_start = 0

        map_16xLxL[i, :, :, :] = constraint_matrix(seq[i])
        for j in range(seq_len):
            seq_0123[i, j] = int(cur_seq[j - seq_start])
            if j < seq_start:
                seq_one_hot[i, j, :] = 0.25
            else:
                try:
                    seq_one_hot[i, j, int(cur_seq[j - seq_start])] = 1
                except:
                    seq_one_hot[i, j, :] = 0.25
    return seq_one_hot, map_16xLxL, label


def data_loading(filename):
    seq01, map_16xLxL, label = seq_to_one_hot_map(filename)
    train_data = []
    label = np.vstack(label)
    for i in range(label.shape[0]):
        train_data.append((seq01[i], map_16xLxL[i], label[i]))

    print("---------------------  Loading Data Completed  -------------------")
    return train_data

