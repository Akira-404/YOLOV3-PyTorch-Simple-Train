import argparse
import yaml

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YOLO
from nets.yolo_loss import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes
from utils.utils_fit import fit_one_epoch


# 加载配置文件
def load_conf(conf_path: str) -> dict:
    """
    :param conf_path :type:yaml
    :return: :type:dict yolo config
    """
    with open(conf_path, 'r') as f:
        conf = yaml.safe_load(f)
    print(conf)
    return conf


conf = load_conf('yolo_config.yaml')
CUDA = True if (torch.cuda.is_available() and conf["cuda"]) else False
device = torch.device('cuda' if CUDA else 'cpu')
print(f'CUDA:{CUDA}')


# 加载权重
def load_weights(model, model_path, device):
    print(f'Load weights {model_path}')
    model_dict = model.state_dict()
    _model_dict = {}
    pretrained_dict = torch.load(model_path, map_location=device)

    for k,v in model_dict.items():
        # pytorch 0.4.0后BN layer新增 num_batches_tracked 参数
        if 'num_batches_tracked' in k:
            print('pass->', k)
        else:
            _model_dict[k] = v
    load_dict = {}
    for kv1, kv2 in zip(_model_dict.items(), pretrained_dict.items()):
        if np.shape(kv1[1]) == np.shape(kv2[1]):
            load_dict[kv1[0]] = kv2[1]
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)


def freeze_train(model, yolo_conf, model_train, train_lines, val_lines, num_classes, num_train, num_val, yolo_loss,
                 loss_history):
    batch_size = yolo_conf['Freeze_batch_size']
    lr = eval(yolo_conf['Freeze_lr'])
    start_epoch = yolo_conf['Init_Epoch']
    end_epoch = yolo_conf['Freeze_Epoch']

    optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    train_dataset = YoloDataset(train_lines, yolo_conf['input_shape'], num_classes, train=True)
    val_dataset = YoloDataset(val_lines, yolo_conf['input_shape'], num_classes, train=False)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=yolo_conf['num_workers'],
                     pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=yolo_conf['num_workers'],
                         pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    if yolo_conf['Freeze_Train']:
        for param in model.backbone.parameters():
            param.requires_grad = False

    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                      epoch_step, epoch_step_val, gen, gen_val, end_epoch, CUDA)
        lr_scheduler.step()


def unfreeze_train(model, yolo_conf, model_train, train_lines, val_lines, num_classes, num_train, num_val, yolo_loss,
                   loss_history):
    batch_size = yolo_conf['Unfreeze_batch_size']
    lr = eval(yolo_conf['Unfreeze_lr'])
    start_epoch = yolo_conf['Freeze_Epoch']
    end_epoch = yolo_conf['UnFreeze_Epoch']

    optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    train_dataset = YoloDataset(train_lines, yolo_conf['input_shape'], num_classes, train=True)
    val_dataset = YoloDataset(val_lines, yolo_conf['input_shape'], num_classes, train=False)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=yolo_conf['num_workers'],
                     pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=yolo_conf['num_workers'],
                         pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    if yolo_conf['Freeze_Train']:
        for param in model.backbone.parameters():
            param.requires_grad = True

    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                      epoch_step, epoch_step_val, gen, gen_val, end_epoch, CUDA)
        lr_scheduler.step()


def train(yolo_conf: dict):
    class_names, num_classes = get_classes(yolo_conf['classes_path'])
    anchors, num_anchors = get_anchors(yolo_conf['anchors_path'])
    print(f'classes_names:{class_names}\nnum_classes:{num_anchors}\n')
    print(f'anchors:{anchors}\nnum_anchors:{num_anchors}')

    model = YOLO(yolo_conf['anchors_mask'], num_classes)
    weights_init(model)
    print('YOLOV3 Weights Init Done.')

    # 载入yolo weight
    if yolo_conf['model_path'] != '':
        load_weights(model, yolo_conf['model_path'], device)
        # print(f'Load weights {yolo_conf["model_path"]}')
        # model_dict = model.state_dict()
        #
        # print('my yolov3 weights key')
        # for k, v in model_dict.items():
        #     print(k)
        # pretrained_dict = torch.load(yolo_conf["model_path"], map_location=device)
        #
        # print('yolov3 weights key')
        # for k, v in pretrained_dict.items():
        #     print(k)
        #
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
    model_train = model.train()

    if CUDA:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss = YOLOLoss(anchors, num_classes, yolo_conf['input_shape'], CUDA, yolo_conf['anchors_mask'])
    loss_history = LossHistory("logs/")

    # load train val dataset txt
    with open(yolo_conf['train_annotation_path']) as f:
        train_lines = f.readlines()
    with open(yolo_conf['val_annotation_path']) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # ----------------------------------------------------------------
    # 冻结训练
    freeze_train(model, yolo_conf, model_train, train_lines, val_lines, num_classes, num_train, num_val, yolo_loss,
                 loss_history)
    # batch_size = yolo_conf['Freeze_batch_size']
    # lr = yolo_conf['Freeze_lr']
    # start_epoch = yolo_conf['Init_Epoch']
    # end_epoch = yolo_conf['Freeze_Epoch']
    #
    # optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
    #
    # train_dataset = YoloDataset(train_lines, yolo_conf['input_shape'], num_classes, train=True)
    # val_dataset = YoloDataset(val_lines, yolo_conf['input_shape'], num_classes, train=False)
    # gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=yolo_conf['num_workers'],
    #                  pin_memory=True,
    #                  drop_last=True, collate_fn=yolo_dataset_collate)
    # gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=yolo_conf['num_workers'],
    #                      pin_memory=True,
    #                      drop_last=True, collate_fn=yolo_dataset_collate)
    #
    # epoch_step = num_train // batch_size
    # epoch_step_val = num_val // batch_size
    #
    # if epoch_step == 0 or epoch_step_val == 0:
    #     raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
    #
    # if yolo_conf['Freeze_Train']:
    #     for param in model.backbone.parameters():
    #         param.requires_grad = False
    #
    # for epoch in range(start_epoch, end_epoch):
    #     fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
    #                   epoch_step, epoch_step_val, gen, gen_val, end_epoch, CUDA)
    #     lr_scheduler.step()

    # --------------------------------------------------------------
    # 全网络训练

    unfreeze_train(model, yolo_conf, model_train, train_lines, val_lines, num_classes, num_train, num_val, yolo_loss,
                   loss_history)
    # batch_size = yolo_conf['Unfreeze_batch_size']
    # lr = yolo_conf['Unfreeze_lr']
    # start_epoch = yolo_conf['Freeze_Epoch']
    # end_epoch = yolo_conf['UnFreeze_Epoch']
    #
    # optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
    #
    # train_dataset = YoloDataset(train_lines, yolo_conf['input_shape'], num_classes, train=True)
    # val_dataset = YoloDataset(val_lines, yolo_conf['input_shape'], num_classes, train=False)
    # gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=yolo_conf['num_workers'],
    #                  pin_memory=True,
    #                  drop_last=True, collate_fn=yolo_dataset_collate)
    # gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=yolo_conf['num_workers'],
    #                      pin_memory=True,
    #                      drop_last=True, collate_fn=yolo_dataset_collate)
    #
    # epoch_step = num_train // batch_size
    # epoch_step_val = num_val // batch_size
    #
    # if epoch_step == 0 or epoch_step_val == 0:
    #     raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
    #
    # if yolo_conf['Freeze_Train']:
    #     for param in model.backbone.parameters():
    #         param.requires_grad = True
    #
    # for epoch in range(start_epoch, end_epoch):
    #     fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
    #                   epoch_step, epoch_step_val, gen, gen_val, end_epoch, CUDA)
    #     lr_scheduler.step()


def main():
    train(conf)


if __name__ == '__main__':
    main()
