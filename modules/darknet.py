import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class BasicConv(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 kernel_size=(1, 1),
                 stride=(1, 1),
                 padding=0,
                 bias: bool = False,
                 act: str = 'leaky_relu'):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(input_channel,
                              output_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.bn = nn.BatchNorm2d(output_channel)

        if act == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1818)
        elif act == 'mish':
            # self.activation = Mish()
            self.activation = nn.Mish(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out


# Res module
class ResBlock(nn.Module):
    def __init__(self, input_channel: int, output_channel: list, act: str):
        """
        note:
        input->1x1 conv -> 3x3 conv->output1
        output1+input->output2
        input channel==output channel

        :param input_channel: type:int one number
        :param output_channel: type:list[int] list number eg:[n1,n2]
        """
        super(ResBlock, self).__init__()
        # 1x1
        # self.conv1 = nn.Conv2d(input_channel, output_channel[0], kernel_size=(1, 1), stride=(1, 1), padding=0,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(output_channel[0])
        # self.leaky_relu1 = nn.LeakyReLU(0.1)

        # 3x3
        # self.conv2 = nn.Conv2d(output_channel[0], output_channel[1], kernel_size=(3, 3), stride=(1, 1), padding=1,
        #                        bias=False)
        # self.bn2 = nn.BatchNorm2d(output_channel[1])
        # self.leaky_relu2 = nn.LeakyReLU(0.1)
        self.conv1 = BasicConv(input_channel, output_channel[0], kernel_size=(1, 1), stride=(1, 1), padding=0,
                               bias=False, act=act)
        self.conv2 = BasicConv(output_channel[0], output_channel[1], kernel_size=(3, 3), stride=(1, 1), padding=1,
                               bias=False, act=act)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += residual

        return out


class DarkNet(nn.Module):
    def __init__(self, layers: list, activation: str):
        super(DarkNet, self).__init__()
        self.input_channel = 32

        # (416,416,3)->(416,416,32)
        self.conv = BasicConv(3, self.input_channel, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False,
                              act=activation)

        # 416,416,32 -> 208,208,64
        self.layer1 = self._make_res_layer([32, 64], layers[0], activation)  # 1
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_res_layer([64, 128], layers[1], activation)  # 2
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_res_layer([128, 256], layers[2], activation)  # 8
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_res_layer([256, 512], layers[3], activation)  # 8
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_res_layer([512, 1024], layers[4], activation)  # 4

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_res_layer(self, io_channels: list, blocks_num: int, act: str):
        """
        note:
        create ResX=Conv+nxRes unit
        :param io_channels: type:list[int],Resblock [input,output]
        :param blocks_num: type:int num of Resblock
        :return:
        """

        # down sampling stride=2
        layers = [('conv', BasicConv(self.input_channel,
                                     io_channels[1],
                                     kernel_size=(3, 3),
                                     stride=(2, 2), padding=1,
                                     bias=False,
                                     act=act))]

        # this input_channel is changed
        self.input_channel = io_channels[1]
        for i in range(0, blocks_num):
            layers.append(('residual_{}'.format(i), ResBlock(self.input_channel, io_channels, act)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out1 = self.layer3(x)  # 52x52
        out2 = self.layer4(out1)  # 26x26
        out3 = self.layer5(out2)  # 13x13

        return out1, out2, out3


def darknet53(activation: str):
    _support_activation = ['leaky_relu', 'mish']
    activation = 'leaky_relu' if activation not in _support_activation else activation
    model = DarkNet([1, 2, 8, 8, 4], activation)
    return model


if __name__ == '__main__':
    darknet = darknet53('mish')
    for k, v in darknet.state_dict().items():
        print(k, np.shape(k))
