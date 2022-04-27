import os
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
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils_fit import fit_one_epoch
from utils.utils import get_anchors, get_classes, load_yaml_conf, get_lr_scheduler, set_optimizer_lr, load_weights

exit()


# 加载权重
# def load_weights(model, model_path: str, device, ignore_track: bool = False):
#     model_dict = model.state_dict()
#     _model_dict = {}
#     pretrained_dict = torch.load(model_path, map_location=device)
#     for k, v in model_dict.items():
#
#         # pytorch 0.4.0后BN layer新增 num_batches_tracked 参数
#         # ignore_track=False:加载net中的 num_batches_tracked参数
#         # ignore_track=True:忽略加载net中的 num_batches_tracked参数
#         if 'num_batches_tracked' in k and ignore_track:
#             logger.info('pass item:', k)
#         else:
#             _model_dict[k] = v
#
#     load_dict = {}
#     cnt = 0
#     pretrained_dict = pretrained_dict['model'] if 'model' in pretrained_dict.keys() else pretrained_dict
#     for kv1, kv2 in zip(_model_dict.items(), pretrained_dict.items()):
#         if np.shape(kv1[1]) == np.shape(kv2[1]):
#             load_dict[kv1[0]] = kv2[1]
#             cnt += 1
#
#     model_dict.update(load_dict)
#     model.load_state_dict(model_dict)
#     logger.info(f'Load weight data:{cnt}/{len(pretrained_dict)}')


def train(args):
    logger.info(f'Input config yaml: {args.config}')
    conf = load_yaml_conf(args.config)
    CUDA = True if (torch.cuda.is_available() and conf["cuda"]) else False
    device = torch.device('cuda' if CUDA else 'cpu')
    logger.info(f'CUDA:{CUDA}')
    logger.info(f'Device:{device}')

    local_path = os.path.dirname(__file__)
    classes_path = os.path.join(local_path, conf['classes_path'])
    anchors_path = os.path.join(local_path, conf['anchors_path'])
    logger.info(f'Classes path:{classes_path}')
    logger.info(f'Anchors path:{anchors_path}')

    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    logger.info(f'Classes names:{class_names}')
    logger.info(f'Anchors:{anchors}')
    logger.info(f'Num Workers:{conf["num_workers"]}')
    logger.info(f'SPP args:{conf["spp"]}')
    logger.info(f'Mosaic:{conf["mosaic"]}')
    logger.info(f'Activation:{conf["activation"]}')

    logger.info(f'Freeze_Train:{conf["Freeze_Train"]}')
    if conf['Freeze_Train']:
        logger.info(f'Freeze epoch:{conf["Freeze_Epoch"]}')
        logger.info(f'Freeze batch size:{conf["Freeze_batch_size"]}')
        logger.info(f'Freeze_lr:{eval(conf["Freeze_lr"])}')

    logger.info(f'Unfreeze epoch:{conf["UnFreeze_Epoch"]}')
    logger.info(f'Unfreeze batch size:{conf["Unfreeze_batch_size"]}')
    logger.info(f'Unfreeze_lr:{eval(conf["Unfreeze_lr"])}')

    model = YOLO(conf['anchors_mask'], num_classes, conf['activation'])
    if conf['spp']:
        model = YOLOSPP(conf['anchors_mask'], num_classes, conf['spp'], conf['activation'])

    weights_init(model)
    logger.info('YOLOV3 Weights Init Done.')

    # 载入yolo weight
    if conf['model_path'] != '':
        model_path = os.path.join(local_path, conf["model_path"])
        logger.info(f'Loading weights:{model_path}')
        load_weights(model, model_path, device)
        logger.info('Loading weights done.')
    model_train = model.train()

    if CUDA:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss = YOLOLoss(anchors,
                         num_classes,
                         conf['input_shape'],
                         CUDA,
                         conf['anchors_mask'])

    loss_history = LossHistory("logs/", model, input_shape=conf['input_shape'])

    # load train val dataset txt
    train_file = os.path.join(local_path, conf['train_file'])
    val_file = os.path.join(local_path, conf['val_file'])
    logger.info(f'train file:{train_file}')
    logger.info(f'val file:{val_file}')

    with open(train_file) as f:
        train_lines = f.readlines()
    with open(val_file) as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)
    logger.info(f'num_train:{num_train}')
    logger.info(f'num_val:{num_val}')

    batch_size = conf['Freeze_batch_size'] if conf['Freeze_Train'] else conf['Unfreeze_batch_size']
    logger.info(f'batch_size:{batch_size}')

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

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    logger.info(f'epoch step:{epoch_step}')
    logger.info(f'epoch step val:{epoch_step_val}')

    logger.info(f'Create Dataset...')

    train_dataset = YoloDataset(train_lines, conf['input_shape'], num_classes, train=True, mosaic=conf['mosaic'])
    val_dataset = YoloDataset(val_lines, conf['input_shape'], num_classes, train=False, mosaic=False)

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  num_workers=conf['num_workers'],
                                  pin_memory=True,
                                  drop_last=True,
                                  collate_fn=yolo_dataset_collate)

    val_dataloader = DataLoader(val_dataset,
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=conf['num_workers'],
                                pin_memory=True,
                                drop_last=True,
                                collate_fn=yolo_dataset_collate)
    unfreeze_flag = False
    save_period = 1
    logger.info(f'Begin to train...')

    # 从初始epoch到最后一个epoch
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
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            train_dataloader = DataLoader(train_dataset,
                                          shuffle=True,
                                          batch_size=batch_size,
                                          num_workers=conf['num_workers'],
                                          pin_memory=True,
                                          drop_last=True, collate_fn=yolo_dataset_collate)
            val_dataloader = DataLoader(val_dataset,
                                        shuffle=True,
                                        batch_size=batch_size,
                                        num_workers=conf['num_workers'],
                                        pin_memory=True,
                                        drop_last=True,
                                        collate_fn=yolo_dataset_collate)

            unfreeze_flag = True

        set_optimizer_lr(optimizer, lr_scheduler_func, curr_epoch)

        fit_one_epoch(model_train,
                      model,
                      yolo_loss,
                      loss_history,
                      optimizer,
                      curr_epoch,
                      epoch_step,
                      epoch_step_val,
                      train_dataloader,
                      val_dataloader,
                      conf['UnFreeze_Epoch'],
                      CUDA,
                      save_period)

    loss_history.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('YOLOV3 config.')
    parser.add_argument('--config', '-c', default='cfg/reflective.yaml', type=str,
                        help='training config yaml. eg: person.yaml')
    args = parser.parse_args()
    train(args)
