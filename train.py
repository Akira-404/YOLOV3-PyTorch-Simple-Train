import os
import platform
import argparse

from loguru import logger
import torch
import torch.backends.cudnn as cudnn

import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from modules.yolo import YOLO
from modules.yolo_spp import YOLOSPP
from modules.loss import YOLOLoss, weights_init

from utils.callbacks import LossHistory
from vocdataset import VOCDataset, yolo_dataset_collate
from utils.utils_fit import fit_one_epoch
from utils.utils import get_anchors, get_classes, load_yaml, get_lr_scheduler, set_optimizer_lr, load_weights


def config_info(args):
    logger.info(f'[Train]::Input config yaml: {args.config}')
    conf = load_yaml(args.config)

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
    # logger.info(f'[Train]::Mosaic:{conf["mosaic"]}')
    logger.info(f'[Train]::Activation:{conf["activation"]}')

    logger.info(f'[Train]::Freeze_Train:{conf["Freeze_Train"]}')
    if conf['Freeze_Train']:
        logger.info(f'[Train]::Freeze epoch:{conf["Freeze_Epoch"]}')
        logger.info(f'[Train]::Freeze batch size:{conf["Freeze_batch_size"]}')
        logger.info(f'[Train]::Freeze_lr:{eval(conf["Freeze_lr"])}')

    logger.info(f'[Train]::Unfreeze epoch:{conf["UnFreeze_Epoch"]}')
    logger.info(f'[Train]::Unfreeze batch size:{conf["Unfreeze_batch_size"]}')
    logger.info(f'[Train]::Unfreeze_lr:{eval(conf["Unfreeze_lr"])}')

    return cuda, device, class_names, num_classes, anchors, num_anchors


def train(args):
    cuda, device, class_names, num_classes, anchors, num_anchors = config_info(args)
    conf = load_yaml(args.config)
    cwd = os.path.dirname(__file__)

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
    if conf['model_path'] != '':
        model_path = os.path.join(cwd, conf["model_path"])
        logger.info(f'Loading weights:{model_path}')
        load_weights(model, model_path, device)
        logger.info('Loading weights done.')
    # <<< 载入yolo weight  <<<

    # <<< prepare dataset  <<<
    model_train = model.train()

    if cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    train_file = os.path.join(conf['train_dataset_root'], 'ImageSets/Main', 'trainval.txt')
    with open(train_file) as f:
        train_lines = f.readlines()

    num_train = len(train_lines)
    batch_size = conf['Freeze_batch_size'] if conf['Freeze_Train'] else conf['Unfreeze_batch_size']
    epoch_step = num_train // batch_size

    logger.info(f'[Train]::train file:{train_file}')
    logger.info(f'[Train]::num_train:{num_train}')
    logger.info(f'[Train]::batch_size:{batch_size}')
    logger.info(f'[Train]::epoch step:{epoch_step}')

    my_train_dataset = VOCDataset(args.config)

    nw = 0 if platform.system() != 'Linux' else conf['num_workers']

    train_dataloader = DataLoader(my_train_dataset,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  num_workers=nw,
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

    lr_scheduler_func = get_lr_scheduler(conf['lr_decay_type'], init_lr_fit, min_lr_fit, conf['UnFreeze_Epoch'])
    # <<< init learning rate <<<

    # <<< model loss <<<
    yolo_loss = YOLOLoss(anchors,
                         num_classes,
                         conf['image_shape'],
                         cuda,
                         conf['anchors_mask'])

    loss_history = LossHistory("logs/")
    # <<< model loss <<<

    # <<< begin training model <<<
    unfreeze_flag = False
    save_period = 1
    logger.info(f'Begin to train...')

    for curr_epoch in range(conf['Init_Epoch'], conf['UnFreeze_Epoch']):
        #   如果模型有冻结学习部分
        #   则解冻，并设置参数
        if curr_epoch >= conf['Freeze_Epoch'] and not unfreeze_flag and conf['Freeze_Train']:
            batch_size = conf['Unfreeze_batch_size']
            logger.info(f'Unfreeze train...')
            nbs = 64
            min_lr = eval(conf['Init_lr']) * 0.01
            init_lr_fit = max(batch_size / nbs * eval(conf['Init_lr']), 1e-4)
            min_lr_fit = max(batch_size / nbs * min_lr, 1e-6)

            lr_scheduler_func = get_lr_scheduler(conf['lr_decay_type'], init_lr_fit, min_lr_fit, conf['UnFreeze_Epoch'])

            for param in model.backbone.parameters():
                param.requires_grad = True

            epoch_step = num_train // batch_size
            # epoch_step_val = num_val // batch_size

            if epoch_step == 0:
                # or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            train_dataloader = DataLoader(my_train_dataset,
                                          shuffle=True,
                                          batch_size=batch_size,
                                          num_workers=conf['num_workers'],
                                          pin_memory=True,
                                          drop_last=True, collate_fn=yolo_dataset_collate)

            unfreeze_flag = True

        set_optimizer_lr(optimizer, lr_scheduler_func, curr_epoch)

        fit_one_epoch(model_train,
                      model,
                      yolo_loss,
                      loss_history,
                      optimizer,
                      curr_epoch,
                      epoch_step,
                      train_dataloader,
                      conf['UnFreeze_Epoch'],
                      cuda,
                      save_period)

    loss_history.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('YOLOV3 config.')
    parser.add_argument('--config', '-c', default='data/voc/config.yaml', type=str,
                        help='training config yaml. eg: person.yaml')
    args = parser.parse_args()
    train(args)
