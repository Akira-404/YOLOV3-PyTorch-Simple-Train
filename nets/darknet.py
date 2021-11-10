import math
from collections import OrderedDict

import numpy as np
import torch.nn as nn


class ConvBNLeakyRelu(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 kernel_size=(1, 1),
                 stride=(1, 1),
                 padding=0,
                 bias: bool = False):
        super(ConvBNLeakyRelu, self).__init__()
        self.conv = nn.Conv2d(input_channel,
                              output_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.bn = nn.BatchNorm2d(output_channel)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.leaky_relu(out)
        return out


# 残差结构
class ResBlock(nn.Module):
    def __init__(self, input_channel: int, output_channel: list):
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

        self.cbl1 = ConvBNLeakyRelu(input_channel, output_channel[0], kernel_size=(1, 1), stride=(1, 1), padding=0,
                                    bias=False)

        # 3x3
        # self.conv2 = nn.Conv2d(output_channel[0], output_channel[1], kernel_size=(3, 3), stride=(1, 1), padding=1,
        #                        bias=False)
        # self.bn2 = nn.BatchNorm2d(output_channel[1])
        # self.leaky_relu2 = nn.LeakyReLU(0.1)

        self.cbl2 = ConvBNLeakyRelu(output_channel[0], output_channel[1], kernel_size=(3, 3), stride=(1, 1), padding=1,
                                    bias=False)

    def forward(self, x):
        residual = x
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.leaky_relu1(out)
        #
        # out = self.conv2(out)
        # out = self.bn1(out)
        # out = self.leaky_relu2(out)

        out = self.cbl1(x)
        out = self.cbl2(out)

        out += residual

        return out


class DarkNet(nn.Module):
    def __init__(self, layers: list):
        super(DarkNet, self).__init__()
        self.input_channel = 32

        # input size(416,416,3)->conv->(416,416,32)
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.inplanes)
        # self.leaky_relu1 = nn.LeakyReLU(0.1)

        # (416,416,3)->(416,416,32)
        self.cbl = ConvBNLeakyRelu(3, self.input_channel, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)

        # 416,416,32 -> 208,208,64
        self.layer1 = self._make_res_layer([32, 64], layers[0])  # 1
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_res_layer([64, 128], layers[1])  # 2
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_res_layer([128, 256], layers[2])  # 8
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_res_layer([256, 512], layers[3])  # 8
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_res_layer([512, 1024], layers[4])  # 4

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

    def _make_res_layer(self, io_channels: list, blocks_num: int):
        """
        note:
        make layer of ResOperator: 3x3conv(x=2) + nxResBlock

        :param io_channels: type:list[int],resblock input and output channel
        :param blocks_num: type:int num of resblock
        :return:
        """
        # layers = [
        #     ('ds_conv', nn.Conv2d(self.inplanes, planes[1], kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False)),
        #     ('ds_bn', nn.BatchNorm2d(planes[1])),
        #     ('ds_leaky_relu', nn.LeakyReLU(0.1))]

        # down sampling stride=2
        layers = [('cbl',
                   ConvBNLeakyRelu(self.input_channel, io_channels[1], kernel_size=(3, 3), stride=(2, 2), padding=1,
                                   bias=False))]

        # this input_channel is changed
        self.input_channel = io_channels[1]
        for i in range(0, blocks_num):
            layers.append(('residual_{}'.format(i), ResBlock(self.input_channel, io_channels)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.leaky_relu1(x)
        x = self.cbl(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5


def darknet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model


if __name__ == '__main__':
    darknet = darknet53()
    for k, v in darknet.state_dict().items():
        print(k, np.shape(k))
