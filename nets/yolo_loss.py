import torch
import torch.nn as nn
import math
import numpy as np


def clip_by_tensor(t: torch.Tensor, t_min: torch.Tensor, t_max: torch.Tensor) -> torch.Tensor:
    # if t < t_min:
    #     return t_min
    # if t > t_max:
    #     return t_max
    # return t

    t = t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def MSELoss(pred, target):
    return torch.pow(pred - target, 2)


def BCELoss(pred: torch.Tensor, target: torch.Tensor):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(YOLOLoss, self).__init__()

        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask

        self.ignore_threshold = 0.5
        self.cuda = cuda

    def forward(self, l, input, targets=None):
        bs = input.size(0)
        in_h = input.size(1)
        in_w = input.size(2)

        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w

        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


if __name__ == '__main__':
    a = (10 < 20) * 10
    print(a)
