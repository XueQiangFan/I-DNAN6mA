#!/Users/11834/.conda/envs/Pytorch_GPU/python.exe
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File    ：RNAcontactmap -> network_model
@IDE    ：PyCharm
@Author  : Xue-Qiang Fan

=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, input, output):
        super(ResidualBlock, self).__init__()
        self.residualBlock = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=(3, 3), stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(output),
            nn.ELU(),
            nn.Dropout(0.25),

            nn.Conv2d(input, output, kernel_size=(3, 3), stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(output),
            SELayer(output),
            nn.ELU(),
            nn.Dropout(0.25))

    def forward(self, x):
        residual = x
        out = self.residualBlock(x)
        out += residual
        return out


class self_attention(nn.Module):
    def __init__(self, in_dim):
        super(self_attention, self).__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.elu = nn.ELU()
        self.fc11 = nn.Linear(64, 32)
        self.soft = nn.Softmax(dim=2)

    def forward(self, x):
        out1 = self.fc1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out2 = self.elu(out1)
        out3 = self.fc11(out2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out4 = self.soft(out3)
        out5 = out4.permute(0, 3, 1, 2)
        y = x.permute(0, 3, 2, 1)
        out6 = out5 @ y 
        out7 = out6.permute(0, 3, 2, 1)
        out = torch.sum(out7, 2) / out7.shape[2] 
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.basicConv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ELU(inplace=True),
            nn.Dropout(0.25)
        )

    def forward(self, x):
        x = self.basicConv2d(x)
        return x


class VariableLengthFeatureDetector(nn.Module):
    # 32 x 42x 42
    def __init__(self, in_channels):
        super(VariableLengthFeatureDetector, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch7x7dbl_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(32, 32, kernel_size=7, padding=3)

        self.branch5x5_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(32, 32, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(32, 32, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, 32, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7dbl_1(x)
        branch7x7 = self.branch7x7dbl_2(branch7x7)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

        # branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        # 32 x 42 x 42
        # print("branch_pool", branch_pool.shape)

        outputs = [branch1x1, branch7x7, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class Model(nn.Module):
    def __init__(self, feature_map_channel=28):
        super(Model, self).__init__()

        self.init_cnn_layer = nn.Sequential(
            nn.Conv2d(feature_map_channel, out_channels=16, kernel_size=(1, 1), stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.ELU(inplace=True))

        self.init_resnet_layer = nn.Sequential(
            ResidualBlock(16, 16), ResidualBlock(16, 16), ResidualBlock(16, 16), ResidualBlock(16, 16),
            ResidualBlock(16, 16), ResidualBlock(16, 16), ResidualBlock(16, 16), ResidualBlock(16, 16),
            nn.Conv2d(16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),

            ResidualBlock(32, 32), ResidualBlock(32, 32), ResidualBlock(32, 32), ResidualBlock(32, 32),
            ResidualBlock(32, 32), ResidualBlock(32, 32), ResidualBlock(32, 32), ResidualBlock(32, 32),
            nn.Conv2d(32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),

            ResidualBlock(64, 64), ResidualBlock(64, 64), ResidualBlock(64, 64), ResidualBlock(64, 64),
            ResidualBlock(64, 64), ResidualBlock(64, 64), ResidualBlock(64, 64), ResidualBlock(64, 64),
        )
        self.feat_detector_layer = VariableLengthFeatureDetector(64)

        self.self_attention_layer = self_attention(192)  # 通道数

        self.lstm1_layer = nn.LSTM(input_size=196, hidden_size=128, num_layers=2, dropout=0.25,
                                   batch_first=True, bidirectional=True)
        self.lstm2_layer = nn.LSTM(input_size=128 * 2, hidden_size=256, num_layers=2, dropout=0.25,
                                   batch_first=True, bidirectional=True)
        self.lstm3_layer = nn.LSTM(input_size=512, hidden_size=64, num_layers=2, dropout=0.25,
                                   batch_first=True, bidirectional=True)


        self.fc_layer = nn.Sequential(

            nn.Linear(5248, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ELU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(128, 1),
            nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x1, x2):
        out = self.init_cnn_layer(x2)
        out = self.init_resnet_layer(out)
        out = self.feat_detector_layer(out)
        out = self.self_attention_layer(out)
        out = torch.cat((out.permute(0, 2, 1), x1), 2)
        out, (_, _) = self.lstm1_layer(out)
        out, (_, _) = self.lstm2_layer(out)
        out, (_, _) = self.lstm3_layer(out)
        out = torch.squeeze(out, 0).reshape(-1, 41 * 128)
        out = self.fc_layer(out)
        return out


