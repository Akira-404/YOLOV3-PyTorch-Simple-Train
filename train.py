import os
import platform
import argparse

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger

from modules.yolo import YOLO
from modules.yolo_spp import YOLOSPP
from modules.loss import YOLOLoss, weights_init

from utils.history import LossHistory
from vocdataset import VOCDataset, yolo_dataset_collate
from yolodataset import YOLODataset
from utils.fit import fit_one_epoch
from utils.common import get_anchors, get_classes, load_yaml, get_lr_scheduler, set_optimizer_lr, load_weights, \
    set_random_seed


def config_info(opt):
    logger.info(f'[Train]::Input config yaml: {opt.config}')
    conf = load_yaml(opt.config)

    logger.info(f'[Train]::seed:{conf["seed"]}')
    logger.info(f'[Train]::deterministic:{conf["deterministic"]}')
    logger.info(f'[Train]::benchmark:{conf["benchmark"]}')

    cuda = True if (torch.cuda.is_available() and conf["cuda"]) else False
    device = torch.device('cuda' if cuda else 'cpu')
    logger.info(f'[Train]::cuda:{cuda}')
    logger.info(f'[Train]::Device:{device}')

    cwd = os.path.dirname(__file__)
    classes_path = os.path.join(cwd, conf['classes_path'])
    anchors_path = os.path.join(cwd, conf['anchors_path'])
    logger.info(f'[Train]::Classes path:{classes_path}')
    logger.info(f'[Train]::Anchors path:{anchors_path}')

    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    logger.info(f'[Train]::Classes names:{class_names}')
    logger.info(f'[Train]::Anchors:{anchors}')
    logger.info(f'[Train]::Num Workers:{conf["num_workers"]}')
    logger.info(f'[Train]::SPP args:{conf["spp"]}')
    logger.info(f'[Train]::Activation:{conf["activation"]}')

    logger.info(f'[Train]::Freeze_Train:{conf["Freeze_Train"]}')
    if conf['Freeze_Train']:
        logger.info(f'[Train]::Freeze epoch:{conf["Freeze_Epoch"]}')
        logger.info(f'[Train]::Freeze batch size:{conf["Freeze_batch_size"]}')
        logger.info(f'[Train]::Freeze_lr:{eval(conf["Freeze_lr"])}')

    logger.info(f'[Train]::Total epoch:{conf["Total_Epoch"]}')
    logger.info(f'[Train]::batch size:{conf["batch_size"]}')
    logger.info(f'[Train]::lr:{eval(conf["lr"])}')

    return cuda, device, class_names, num_classes, anchors, num_anchors


def train(opt):
    cuda, device, class_names, num_classes, anchors, num_anchors = config_info(opt)
    conf = load_yaml(opt.config)
    cwd = os.path.dirname(__file__)

    set_random_seed(conf['seed'], conf['deterministic'], conf['benchmark'])

    # <<< get the model <<<
    model = YOLO(conf['anchors_mask'], num_classes, conf['activation'])
    if conf['spp']:
        model = YOLOSPP(conf['anchors_mask'], num_classes, conf['spp'], conf['activation'])
    # <<< get the model <<<

    # <<< init model <<<
    logger.info('Init model weights...')
    weights_init(model)
    logger.info('Init done.')
    # <<< init model <<<

    # <<< 载入yolo weight  <<<
    if conf['model_path'] != '' and opt.resume is False:
        model_path = os.path.join(cwd, conf["model_path"])
        logger.info(f'Loading weights:{model_path}')
        load_weights(model, model_path, device)
        logger.info('Loading weights done.')
    # <<< 载入yolo weight  <<<

    # <<< prepare dataset  <<<
    model_train = model.train()

    if cuda:
        model_train = torch.nn.DataParallel(model)
        model_train = model_train.cuda()

    train_file = None
    if conf['voc']:
        train_file = os.path.join(conf['train_dataset_root'], 'ImageSets/Main', 'trainval.txt')
    elif conf['yolo']:
        train_file = os.path.join(conf['train_dataset_root'], 'train.txt')

    with open(train_file) as f:
        train_lines = f.readlines()

    num_train = len(train_lines)
    batch_size = conf['Freeze_batch_size'] if conf['Freeze_Train'] else conf['batch_size']
    epoch_step = num_train // batch_size

    logger.info(f'[Train]::train file:{train_file}')
    logger.info(f'[Train]::num_train:{num_train}')
    logger.info(f'[Train]::batch_size:{batch_size}')
    logger.info(f'[Train]::epoch step:{epoch_step}')

    train_dataset = VOCDataset(opt.config) if conf['voc'] else YOLODataset(opt.config)

    conf['num_workers'] = 0 if platform.system() != 'Linux' else conf['num_workers']

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  num_workers=conf['num_workers'],
                                  pin_memory=True,
                                  drop_last=True,
                                  collate_fn=yolo_dataset_collate)
    # <<< prepare dataset  <<<

    # <<< init learning rate <<<
    nbs = 64
    min_lr = eval(conf['Init_lr']) * 0.01
    init_lr_fit = max(batch_size / nbs * eval(conf['Init_lr']), 1e-4)
    min_lr_fit = max(batch_size / nbs * min_lr, 1e-6)
    logger.info(f'init_lr_fit:{init_lr_fit}')
    logger.info(f'min_lr_fit:{min_lr_fit}')

    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)

    optimizer = {
        'adam': optim.Adam(pg0, init_lr_fit, betas=(conf['momentum'], 0.999)),
        'sgd': optim.SGD(pg0, init_lr_fit, momentum=conf['momentum'], nesterov=True)
    }[conf['optimizer_type']]

    optimizer.add_param_group({"params": pg1, "weight_decay": eval(conf['weight_decay'])})
    optimizer.add_param_group({"params": pg2})

    lr_scheduler_func = get_lr_scheduler(conf['lr_decay_type'], init_lr_fit, min_lr_fit, conf['Total_Epoch'])
    # <<< init learning rate <<<

    # <<< checkpoint <<<
    if opt.resume != '':
        logger.info('loading checkpoint...')
        # path_checkpoint = os.path.join(cwd, conf["model_path"])  # 断点路径
        checkpoint = torch.load(opt.resume)  # 加载断点

        logger.info('loading checkpoint.state_dict...')
        model.load_state_dict(checkpoint['state_dict'])  # 加载模型可学习参数

        logger.info('loading checkpoint.optimizer...')
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数

        logger.info('loading checkpoint.epoch...')
        conf['Init_Epoch'] = checkpoint['epoch']  # 设置开始的epoch

        logger.info('loading checkpoint done.')
    # <<< checkpoint <<<

    # <<< model loss <<<
    yolo_loss = YOLOLoss(anchors,
                         num_classes,
                         conf['image_shape'],
                         cuda,
                         conf['anchors_mask'])

    loss_history = LossHistory("logs/")
    # <<< model loss <<<

    # <<< begin training model <<<
    unfreeze_flag = False  # use to init net args,just only once.
    save_period = 1
    logger.info(f'Begin to train...')

    # from begin to end
    for curr_epoch in range(conf['Init_Epoch'], conf['Total_Epoch']):

        if curr_epoch >= conf['Freeze_Epoch'] and not unfreeze_flag and conf['Freeze_Train']:
            batch_size = conf['batch_size']
            logger.info(f'Fully network train...')

            # <<< learning rate setting
            nbs = 64
            min_lr = eval(conf['Init_lr']) * 0.01
            init_lr_fit = max(batch_size / nbs * eval(conf['Init_lr']), 1e-4)
            min_lr_fit = max(batch_size / nbs * min_lr, 1e-6)

            lr_scheduler_func = get_lr_scheduler(conf['lr_decay_type'], init_lr_fit, min_lr_fit, conf['Total_Epoch'])

            for param in model.backbone.parameters():
                param.requires_grad = True
            # <<< learning rate setting

            epoch_step = num_train // batch_size

            if epoch_step == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            train_dataloader = DataLoader(train_dataset,
                                          shuffle=True,
                                          batch_size=batch_size,
                                          num_workers=conf['num_workers'],
                                          pin_memory=True,
                                          drop_last=True, collate_fn=yolo_dataset_collate)

            unfreeze_flag = True

        set_optimizer_lr(optimizer, lr_scheduler_func, curr_epoch)

        fit_one_epoch(model,
                      model_train,
                      train_dataloader,
                      yolo_loss,
                      loss_history,
                      optimizer,
                      curr_epoch,
                      epoch_step,
                      conf['Total_Epoch'],
                      cuda,
                      save_period)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('YOLOV3 config.')
    parser.add_argument('--config', '-c', default='data/mask/config.yaml', type=str,
                        help='training config yaml. eg: data/voc/config.yaml')
    parser.add_argument('--resume', '-r', default='logs/checkpoint/ckpt_ep3_loss0.47645998338483414.pth', type=str,
                        help='xxx/xxx/checkpoint.pth')
    args = parser.parse_args()
    train(args)
