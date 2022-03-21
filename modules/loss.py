import torch
import torch.nn as nn
import torch.cuda
import math
import numpy as np
from utils.utils_loss import *
from loguru import logger

class YOLOLoss(nn.Module):
    def __init__(self,
                 anchors: list,
                 num_classes: int,
                 input_shape: list,
                 cuda: bool,
                 anchors_mask: list, ):
        super(YOLOLoss, self).__init__()

        # yolov3 anchors
        # [10,13,  16,30,  33,23,30,61,  62,45,  59,119,116,90,  156,198,  373,326]
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes  # (x,y,w,h,p)+num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask if not anchors_mask else [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.ignore_threshold = 0.5
        self.cuda = cuda
        self.giou = True
        self.balance        = [0.4, 1.0, 4]
        self.box_ratio      = 0.05
        self.obj_ratio      = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.cls_ratio      = 1 * (num_classes / 80)
        

    def forward(self, l, input, targets=None):
        # l:当前第几个有效特征层
        #   input的shape为  bs(batch size),  3*(5+num_classes), 13, 13
        #                  bs,              3*(5+num_classes), 26, 26
        #                  bs,              3*(5+num_classes), 52, 52

        # target shape:batch_size num_boj xmin,ymin,xmax,ymax,classes_id

        bs = input.size(0)
        # in_h in_w为feature layer output shape
        in_h = input.size(2)
        in_w = input.size(3)

        # 计算当前层累积到的缩放尺度
        # layer shape(c,13,13):stride_h(w)=32
        # layer shape(c,26,26):stride_h(w)=16
        # layer shape(c,52,52):stride_h(w)=8
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w

        # 计算anchors在当前层的缩放值
        scaled_anchors = [(anchor_w / stride_w, anchor_h / stride_h) for anchor_w, anchor_h in self.anchors]
        # attributes of a bounding box : (tx,ty,tw,th,pc,c1,c2...cn)x3
        # prediction shape:batch_size,3,13,13,5+num_classes,
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4,
                                                                                                    2).contiguous()
        # 获取调整参数
        '''
        eg:
            a=[[1,2,3],[4,5,6]]
            a.shape=(2.3)              
            a[:,1]=a[...,1]
            
            list[...,idx]=list[;,;,idx]
        '''
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])

        w = prediction[..., 2]
        h = prediction[..., 3]

        conf = torch.sigmoid(prediction[..., 4])
        # 置信度对应的class
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # 获得网络预测结果
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)

        # 判断预测结果和真实值的重合程度
        noobj_mask, pred_boxes = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)

        if self.cuda:
            y_true = y_true.cuda()
            noobj_mask = noobj_mask.cuda()
            box_loss_scale = box_loss_scale.cuda()

        #   reshape_y_true[...,2:3]和reshape_y_true[...,3:4]
        #   表示真实框的宽高，二者均在0-1之间
        #   真实框越大，比重越小，小框的比重更大。
        box_loss_scale = 2 - box_loss_scale
        loss = 0
        obj_mask = y_true[..., 4] == 1
        n = torch.sum(obj_mask)

        if n != 0:
            if self.giou:

                giou = box_giou(pred_boxes, y_true[..., :4])
                loss_loc = torch.mean((1 - giou)[obj_mask])
            else:

                loss_x = torch.mean(BCELoss(x[obj_mask], y_true[..., 0][obj_mask]) * box_loss_scale)
                loss_y = torch.mean(BCELoss(y[obj_mask], y_true[..., 1][obj_mask]) * box_loss_scale)

                loss_w = torch.mean(MSELoss(w[obj_mask], y_true[..., 2][obj_mask]) * box_loss_scale)
                loss_h = torch.mean(MSELoss(h[obj_mask], y_true[..., 3][obj_mask]) * box_loss_scale)
                loss_loc = (loss_x + loss_y + loss_h + loss_w) * 0.1

            loss_cls = torch.mean(BCELoss(pred_cls[obj_mask], y_true[..., 5:][obj_mask]))
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio

        loss_conf = torch.mean(BCELoss(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])
        loss += loss_conf * self.balance[l] * self.obj_ratio
        # if n != 0:
        #     logger.info(loss_loc * self.box_ratio, loss_cls * self.cls_ratio, loss_conf * self.balance[l] * self.obj_ratio)
        return loss

    def get_target(self, l, targets, anchors, in_h, in_w):
        """
        targets shape:(batch_size,obj_size,5) :batch_size num_obj xmin,ymin,xmax,ymax,class_id
        """
        # 计算一共有多少张图片
        bs = len(targets)

        # 用于选取哪些先验框不包含物体
        # noobj_mask shape: batch_size 3,13,13
        noobj_mask = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)

        # 让网络更加去关注小目标
        # box_loss_scale shape: batch_size 3,13,13
        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)

        # batch_size, 3, 13, 13, 5 + num_classes
        y_true = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad=False)
        for b in range(bs):

            if len(targets[b]) == 0:  # 无标注物体
                continue

            # batch_target shape:(num_obj,5): num_obj xmin ymin xmax ymax class_id
            batch_target = torch.zeros_like(targets[b])

            #   计算出正样本在特征层上的中心点
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w  # xmin ymin * in_w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h  # xmax ymax * in_h
            # get obj class
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()

            # 将真实框转换一个形式
            # batch_target.size(0)=num_obj
            # cat:[num_obj,2]+[num_obj,2]->[num_obj,4]
            gt_box = torch.FloatTensor(
                torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), dim=1))

            #   将先验框转换一个形式
            # cat:[9,2]+ [9,2]=[9,4]
            anchor_shapes = torch.FloatTensor(
                torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), dim=1))

            #   计算交并比
            #   self.calculate_iou(gt_box, anchor_shapes) = [num_true_box, 9]每一个真实框和9个先验框的重合情况
            #   best_ns:
            #   [每个真实框最大的重合度max_iou, 每一个真实框最重合的先验框的序号]
            best_ns = torch.argmax(calculate_iou(gt_box, anchor_shapes), dim=-1)  # dim=-1:最后一个维度

            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[l]:
                    continue

                #   判断这个先验框是当前特征点的哪一个先验框
                k = self.anchors_mask[l].index(best_n)

                #   获得真实框属于哪个网格点
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()

                #   取出真实框的种类
                c = batch_target[t, 4].long()

                #   noobj_mask代表无目标的特征点
                # noobj_mask shape: batch_size 3,13,13
                noobj_mask[b, k, j, i] = 0

                #   tx、ty代表中心调整参数的真实值
                y_true[b, k, j, i, 0] = batch_target[t, 0] - i.float()
                y_true[b, k, j, i, 1] = batch_target[t, 1] - j.float()
                y_true[b, k, j, i, 2] = math.log(batch_target[t, 2] / anchors[best_n][0])
                y_true[b, k, j, i, 3] = math.log(batch_target[t, 3] / anchors[best_n][1])
                y_true[b, k, j, i, 4] = 1
                y_true[b, k, j, i, c + 5] = 1

                #   用于获得xywh的比例
                #   大目标loss权重小，小目标loss权重大
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h

        return y_true, noobj_mask, box_loss_scale

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        #   计算一共有多少张图片
        bs = len(targets)

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        #   生成网格，先验框中心，网格左上角
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([1]))

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        #   计算调整后的先验框中心与宽高
        pred_boxes_x = torch.unsqueeze(x.data + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y.data + grid_y, -1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w.data) * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h.data) * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)

        for b in range(bs):

            #   将预测结果转换一个形式
            #   pred_boxes_for_ignore.shape:[num_anchors, 4]
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)

            #   计算真实框，并把真实框转换成相对于特征层的大小
            #   gt_box.shape[num_true_box, 4]
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])

                #   计算出正样本在特征层上的中心点
                batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
                batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
                batch_target = batch_target[:, :4]

                #   计算交并比
                #   anch_ious.shape[num_true_box, num_anchors]
                anch_ious = calculate_iou(batch_target, pred_boxes_for_ignore)

                #   每个先验框对应真实框的最大重合度
                #   anch_ious_max.shpae[num_anchors]
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask, pred_boxes


def weights_init(net, init_type: str = 'normal', init_gain: float = 0.02):
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

    logger.info('initialize network with %s type' % init_type)
    net.apply(init_func)
