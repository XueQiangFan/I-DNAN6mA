#!/Users/11834/.conda/envs/Pytorch_GPU/python.exe
# -*- coding: UTF-8 -*-

import torch, os, re, random, gc
import numpy as np
from data_loading import data_loading
from configparser import ConfigParser
from torch.utils.data import DataLoader
from network_model import Model
from Util.WriteFile import appendWrite
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config = ConfigParser()

def main(args):
    saved_models = args.saved_models
    model = Model(16).to(device)
    save_model = saved_models+"model"
    model.load_state_dict(torch.load(save_model, map_location="cpu"))
    optimizer = torch.optim.Adam(model.parameters())
    save_model = saved_models + 'modelopt'
    optimizer.load_state_dict(torch.load(save_model, map_location="cpu"))

    model.eval()
    test_sample = data_loading(args.test_path)
    test_loader = DataLoader(dataset=test_sample, batch_size=380, shuffle=False, drop_last=False)
    print('batch_num [{}]'.format(test_loader.__len__()))
    predict_prob = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            one_hot, one_hot_map = data
            one_hot, one_hot_map = torch.FloatTensor(one_hot.float()).to(device), torch.FloatTensor(one_hot_map.float()).to(device)
            predict = model(one_hot, one_hot_map)
            for j in range(one_hot.shape[0]):
                predict_prob.append(np.round(predict[j].item(), 4))
        predict_prob = np.array(predict_prob)
        appendWrite(args.result_path, '{:>4}\n\n'.format("# I-DNAN6mA VFORMAT (I-DNAN6mA V1.0)"))
        appendWrite(args.result_path, '{:>1}  {:>4}  {:>4}\t\n'.format("NO.", "Prob", "6mA site"))
        for i in range(predict_prob.shape[0]):
            index, prob = i + 1, predict_prob[i]
            if prob >= 0.5:
                appendWrite(args.result_path, '{:>4}  {:>.3f}  {:>.3}\t\n'.format(index, prob, "Yes"))
            else:
                appendWrite(args.result_path, '{:>4}  {:>.3f}  {:>.3}\t\n'.format(index, prob, "NO"))
        appendWrite(args.result_path, '{:>8} \t'.format("END"))


if __name__ == '__main__':
    ######################
    # Adding Arguments
    #####################

    p = argparse.ArgumentParser(description=''' step 1. Features Generation From DNA Sequences
                                                step 2. Run I-DNAN6mA ''')
    p.add_argument('-saved_models', '--saved_models', type=str, help='~/saved_models')
    p.add_argument('-test_path', '--test_path', type=str, help='~/test.csv')
    p.add_argument('-result_path', '--result_path', type=str, help='~/result.csv')
    args = p.parse_args()
    main(args)
