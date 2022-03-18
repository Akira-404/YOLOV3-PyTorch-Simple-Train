import torch
import torch.cuda
import math


def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # ----------------------------------------------------#
    #   求出预测框左上角右下角
    # ----------------------------------------------------#
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # ----------------------------------------------------#
    #   求出真实框左上角右下角
    # ----------------------------------------------------#
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # ----------------------------------------------------#
    #   求真实框和预测框所有的iou
    # ----------------------------------------------------#
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area, min=1e-6)

    # ----------------------------------------------------#
    #   计算中心的差距
    # ----------------------------------------------------#
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

    # ----------------------------------------------------#
    #   找到包裹两个框的最小框的左上角和右下角
    # ----------------------------------------------------#
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    # ----------------------------------------------------#
    #   计算对角线距离
    # ----------------------------------------------------#
    enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
    ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(
        b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
        b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
    alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
    ciou = ciou - alpha * v
    return ciou


def smooth_labels(y_true, label_smoothing, num_classes):
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes


# 限制数据范围
def clip_by_tensor(t: torch.Tensor, t_min, t_max):
    t = t.float()
    # _t = t_min if (t < t_min).float() else t
    # _t = t_max if (t > t_max).float() else _t
    # return _t

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def MSELoss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.pow(pred - target, 2)


def BCELoss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


def calculate_iou(_box_a: torch.Tensor, _box_b: torch.Tensor) -> torch.Tensor:
    """
    _box_a:gt box shape:[num_obj,4]=[num_obj,[cx,cy,w,h]]
    _box_h:anchors box shape:[9,4]=[num_anchors,[cx,cy,w,h]]
    """
    #   计算真实框的左上角和右下角
    # x1=cx-w/2,x2=cx+w/2
    # y1=cy-h/2,y2=cy+h/2

    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    #   计算先验框获得的预测框的左上角和右下角
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

    #   将真实框和预测框都转化成左上角右下角的形式
    box_a = torch.zeros_like(_box_a)  # [num_obj,4]
    box_b = torch.zeros_like(_box_b)  # [num_anchors,4]
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

    #   A为真实框的数量，B为先验框的数量
    A = box_a.size()[0]
    B = box_b.size()[0]
    # A = box_a.size(0)
    # B = box_b.size(0)

    #   计算交的面积
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]
    #   计算预测框和真实框各自的面积
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    #   求IOU
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def box_giou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回为：
    -------
    giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # ----------------------------------------------------#
    #   求出预测框左上角右下角
    # ----------------------------------------------------#
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # ----------------------------------------------------#
    #   求出真实框左上角右下角
    # ----------------------------------------------------#
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # ----------------------------------------------------#
    #   求真实框和预测框所有的iou
    # ----------------------------------------------------#
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / union_area

    # ----------------------------------------------------#
    #   找到包裹两个框的最小框的左上角和右下角
    # ----------------------------------------------------#
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    # ----------------------------------------------------#
    #   计算对角线距离
    # ----------------------------------------------------#
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    giou = iou - (enclose_area - union_area) / enclose_area

    return giou
