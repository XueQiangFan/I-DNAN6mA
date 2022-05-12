#!/Users/11834/.conda/envs/Pytorch_GPU/python.exe
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File    ：RNASolventAccessibility -> Trainer
@IDE    ：PyCharm
@Date   ：2021/1/29 15:53
=================================================='''
import torch, os, re, random, gc
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
# from Fan_AL6mA_one_hot.data_loading import data_loading
from I_DNAN6mA_A.data_loading_ import data_loading
from configparser import ConfigParser
from torch.utils.data import DataLoader
from I_DNAN6mA_A.network_model import Model
from I_DNAN6mA_A.Util.loss_function import MSELoss
from I_DNAN6mA_A.Util.WriteFile import appendWrite
from I_DNAN6mA_A.Util.calculate_confusion_matrix import calculate_TP_FP_TN_FN, calculate_confusion_matrix

config = ConfigParser()
config.read('I_DNAN6msite.config')
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def set_seed(seed=6):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed()
learn_rate = config.getfloat('parameter', 'learn_rate')
batch_size = config.getint('parameter', 'batch_size')
num_epochs = config.getint('parameter', 'num_epochs')

model_folder = config.get('getfile', 'model_folder')
train_loss_path = config.get('getfile', 'train_loss_path')
test_loss_path = config.get('getfile', 'test_loss_path')
thaliana_result_path = config.get('getfile', 'thaliana_result_path')
melanogaster_result_path = config.get('getfile', 'melanogaster_result_path')

train_dir = config.get('dataset', 'train_dir')
test_dir = config.get('dataset', 'test_dir')


def train(model, optimizer, lossfunction, start_epoch=0, num_epochs=5):
    # train_sample, val_sample, test_sample = data_loading(train_dir)
    train_sample = data_loading(train_dir)
    train_loader = DataLoader(dataset=train_sample, batch_size=batch_size, shuffle=True, drop_last=False)
    batch_num = train_loader.__len__()
    print("Total batch number[{}]".format(batch_num))
    print('start_epoch [{}]'.format(start_epoch))
    print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))
    for epoch in range(start_epoch, num_epochs):
        model.train()
        for i, data in enumerate(train_loader):
            one_hot, one_hot_map, label = data
            one_hot, one_hot_map = torch.FloatTensor(one_hot.float()), torch.FloatTensor(one_hot_map.float())
            one_hot, one_hot_map = Variable(one_hot).to(device), Variable(one_hot_map.float()).to(device)
            label = torch.FloatTensor(label.float())
            label = Variable(label).to(device)
            # print(one_hot.shape, one_hot_map.shape, label.shape)
            predict = model(one_hot, one_hot_map)
            loss = lossfunction(predict, label)
            now_batch_loss = float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del predict
            del loss
            gc.collect()
            if (i + 1) % 1 == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}\n'.format(epoch + 1, num_epochs, i + 1, batch_num,
                                                                            now_batch_loss))
                appendWrite(train_loss_path,
                            'Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}\n'.format(epoch + 1, num_epochs, i + 1,
                                                                                  batch_num, now_batch_loss))

        #if (epoch + 1) >= 1 and (epoch + 1) % 1 == 0:
        #    torch.save(model.state_dict(), model_folder + 'epoch_' + str(epoch + 1), _use_new_zipfile_serialization=False)
        #    torch.save(optimizer.state_dict(), model_folder + 'epoch_' + str(epoch + 1) + 'opt', _use_new_zipfile_serialization=False)

        if (epoch + 1) >= 1 and (epoch + 1) % 1 == 0:
            result_path = './' + config.get('getfile', 'thaliana_result_path')
            #result_path = './' + config.get('getfile', 'melanogaster_result_path')
            test(model, test_dir, lossfunction, result_path, epoch + 1)

        # if (epoch + 1) == 200:
            exit(-1)


def test(model, test_dir, lossfunction, log_path, epoch):
    model.eval()
    test_sample = data_loading(test_dir)
    test_loader = DataLoader(dataset=test_sample, batch_size=50, shuffle=False, drop_last=False)
    print('batch_num [{}]'.format(test_loader.__len__()))
    TP, FP, TN, FN = 0, 0, 0, 0
    total_TP, total_TN = 0, 0
    predict_prob, label_prob = [], []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            one_hot, one_hot_map, label = data
            one_hot, one_hot_map = torch.FloatTensor(one_hot.float()), torch.FloatTensor(one_hot_map.float())
            one_hot, one_hot_map = Variable(one_hot).to(device), Variable(one_hot_map.float()).to(device)
            label = torch.FloatTensor(label.float())
            label = Variable(label).to(device)
            predict = model(one_hot, one_hot_map)
            loss = lossfunction(predict, label)
            test_loss = float(loss)
            print('Test Loss: {:.4f}\n'.format(test_loss))
            appendWrite(test_loss_path, 'Epoch:[{}], Test Loss: {:.4f}\n'.format(epoch, test_loss))
            tp, fp, tn, fn, total_tp, total_tn = calculate_TP_FP_TN_FN(predict, label)
            for i in range(label.shape[0]):
                predict_prob.append(np.round(predict[i].item(), 4))
                label_prob.append(label[i].item())
            TP, FP, TN, FN, total_TP, total_TN = TP + tp, FP + fp, TN + tn, FN + fn, total_TP + total_tp, total_TN + total_tn
        print("total TP:{}, total TN:{},\nTP:{}, FP:{}, TN:{}, FN:{}".format(total_TP, total_TN, TP, FP, TN, FN))
        predict_prob, label_prob = np.array(predict_prob), np.array(label_prob)
        AUROC = roc_auc_score(label_prob, predict_prob)
        Sen, Spe, Pre, Acc, Mcc, FOne = calculate_confusion_matrix(TP, FP, TN, FN)
        # appendWrite(log_path,
        #             'Epoch:{}, total TP:{}, total TN:{}, TP:{}, FP:{}, TN:{}, FN:{}\n'.format(epoch, total_TP, total_TN,
        #                                                                                       TP, FP, TN, FN))
        appendWrite(log_path,
                    'Epoch:{}, Sen:{:.3f}, Spe:{:.3f}, Pre:{:.3f}, Acc:{:.3f}, Mcc:{:.3f}, AUROC:{:.3f}, FOne{:.3f}\n'.format(
                        epoch, Sen, Spe, Pre,
                        Acc, Mcc, AUROC, FOne))
        return label_prob, predict_prob


# if __name__ == '__main__':
def main():
    model = Model(16).to(device)
    loss_function = MSELoss
    # loss_function = torch.nn.BCEWithLogitsLoss
    start_epoch = 0
    if not os.path.isdir(model_folder):
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learn_rate, weight_decay=0.01)
        os.mkdir(model_folder)
    else:
        model_names = os.listdir(model_folder)
        if len(model_names) > 1:
            start_epoch = max([int(re.sub("\D", "", ss)) for ss in model_names])
            saved_model = model_folder + 'epoch_' + str(start_epoch)
            model.load_state_dict(torch.load(saved_model, map_location='cuda:0'))
            optimizer = torch.optim.Adam(model.parameters())
            saved_model = model_folder + 'epoch_' + str(start_epoch) + 'opt'
            optimizer.load_state_dict(torch.load(saved_model, map_location='cuda:0'))
        else:
            start_epoch = 0
            optimizer = torch.optim.Adam(params=model.parameters(), lr=learn_rate, weight_decay=0.01)
    #train(model, optimizer, loss_function, num_epochs=num_epochs, start_epoch=start_epoch)
    result_path = config.get('getfile', 'thaliana_result_path')
    label_prob, predict_prob = test(model, test_dir, loss_function, result_path, 52)
    return label_prob, predict_prob

