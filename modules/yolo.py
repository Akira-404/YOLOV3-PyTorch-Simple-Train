from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from modules.darknet import darknet53


def conv2d(in_channel: int, out_channel: int, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0

    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=(1, 1), padding=pad, bias=False)),
        ('bn', nn.BatchNorm2d(out_channel)),
        ('leaky_relu', nn.LeakyReLU(0.1818))
    ]))


def _make_yolo_block(channels: list, in_channel: int, out_channel: int):
    """

    note: yolo block 2x[1x1conv(c=n)+3x3conv(c=2n)]+1x1conv(c=n)
    :param channels: type:list[int] eg:[channel,2xchannel]
    :param in_channel:
    :param out_channel:
    :return:
    """
    m = nn.Sequential(

        conv2d(in_channel, channels[0], kernel_size=1),
        conv2d(channels[0], channels[1], kernel_size=3),

        conv2d(channels[1], channels[0], kernel_size=1),
        conv2d(channels[0], channels[1], kernel_size=3),

        conv2d(channels[1], channels[0], kernel_size=1),
        conv2d(channels[0], channels[1], kernel_size=3),

        nn.Conv2d(channels[1], out_channel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)
    )
    return m


class SPP(nn.Module):
    def __init__(self, pool_sizes=None):
        super(SPP, self).__init__()

        pool_sizes = [5, 9, 13] if pool_sizes is None else pool_sizes

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


class YOLO(nn.Module):
    def __init__(self, anchors_mask: list, num_classes: int, spp: bool = False, act: str = 'leaky_relu'):
        super(YOLO, self).__init__()

        self.backbone = darknet53(act)

        out_filters = self.backbone.layers_out_filters

        # big object
        self.big_detect_layer = _make_yolo_block([512, 1024],
                                                 out_filters[-1],
                                                 len(anchors_mask[0]) * (num_classes + 5))
        self.spp = SPP() if spp else spp

        # medium object
        self.medium_detect_layer_conv = conv2d(512, 256, 1)
        self.medium_detect_layer_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.medium_detect_layer = _make_yolo_block([256, 512],
                                                    out_filters[-2] + 256,  # concat
                                                    len(anchors_mask[1]) * (num_classes + 5))

        # small object
        self.small_detect_layer_conv = conv2d(256, 128, 1)
        self.small_detect_layer_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.small_detect_layer = _make_yolo_block([128, 256],
                                                   out_filters[-3] + 128,  # concat
                                                   len(anchors_mask[2]) * (num_classes + 5))

    def forward(self, x):
        # small x2:(52,52,256)
        # medium x1:(26,26,512)
        # big x0:(13,13,1024)
        x2, x1, x0 = self.backbone(x)

        # input x0:(13,13,1024) out0_branch:(13,13,512),out0:(13,13,512)
        out0_branch = self.big_detect_layer[:5](x0)  # 上采样位置
        out0 = self.big_detect_layer[5:](out0_branch)
        if self.spp:
            out0 = self.spp(out0)

        else:
            out0 = out0

        # big+medium上采样+数据拼接
        x1_in = self.medium_detect_layer_conv(out0_branch)
        x1_in = self.medium_detect_layer_upsample(x1_in)
        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)

        # input x1_in:(26,26,768) out1_branch:(26,26,512),out0:(26,26,256)
        out1_branch = self.medium_detect_layer[:5](x1_in)  # 上采样位置
        out1 = self.medium_detect_layer[5:](out1_branch)

        # medium+small上采样+数据拼接
        x2_in = self.small_detect_layer_conv(out1_branch)
        x2_in = self.small_detect_layer_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)

        out2 = self.small_detect_layer(x2_in)
        return out0, out1, out2


if __name__ == '__main__':
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    net = YOLO(anchors_mask, 2, spp=True, act='mish')
    with open('yolo.txt', 'w') as f:
        f.write(str(net))
    exit()
    dict = net.state_dict()
    # for k,v in dict.items():
    #     print(k,' ',np.shape(v))

    yw = torch.load('../data/voc.pth', map_location='cpu')
    for k, v in yw.items():
        print(k, ' ', np.shape(v))
