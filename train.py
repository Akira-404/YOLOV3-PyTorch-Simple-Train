import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from modules.yolo import YOLO
from modules.yolo_spp import YOLOSPP
from modules.loss import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes, load_yaml_conf
from utils.utils_fit import fit_one_epoch

conf = load_yaml_conf('train.yaml')
CUDA = True if (torch.cuda.is_available() and conf["cuda"]) else False
device = torch.device('cuda' if CUDA else 'cpu')
print(f'CUDA:{CUDA}')


# 加载权重
def load_weights(model, model_path: str, ignore_track: bool = False):
    model_dict = model.state_dict()
    _model_dict = {}
    pretrained_dict = torch.load(model_path, map_location=device)

    for k, v in model_dict.items():

        # pytorch 0.4.0后BN layer新增 num_batches_tracked 参数
        # ignore_track=False:加载net中的 num_batches_tracked参数
        # ignore_track=True:忽略加载net中的 num_batches_tracked参数
        if 'num_batches_tracked' in k and ignore_track:
            print('pass item:', k)
        else:
            _model_dict[k] = v

    load_dict = {}
    cnt = 0
    pretrained_dict = pretrained_dict['model'] if 'model' in pretrained_dict.keys() else pretrained_dict
    for kv1, kv2 in zip(_model_dict.items(), pretrained_dict.items()):
        if np.shape(kv1[1]) == np.shape(kv2[1]):
            load_dict[kv1[0]] = kv2[1]
            cnt += 1

    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    print(f'Load weight data:{cnt}/{len(pretrained_dict)}')


def freeze_train(model, model_train, train_lines, val_lines, num_classes, num_train, num_val, yolo_loss,
                 loss_history):
    batch_size = conf['Freeze_batch_size']
    lr = eval(conf['Freeze_lr'])
    start_epoch = conf['Init_Epoch']
    end_epoch = conf['Freeze_Epoch']
    mosaic = conf['mosaic']

    optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
    if conf['cosine_lr']:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    train_dataset = YoloDataset(train_lines, conf['input_shape'], num_classes, train=True, mosaic=mosaic)
    val_dataset = YoloDataset(val_lines, conf['input_shape'], num_classes, train=False, mosaic=False)

    gen = DataLoader(train_dataset,
                     shuffle=True,
                     batch_size=batch_size,
                     num_workers=conf['num_workers'],
                     pin_memory=True,
                     drop_last=True,
                     collate_fn=yolo_dataset_collate)

    gen_val = DataLoader(val_dataset,
                         shuffle=True,
                         batch_size=batch_size,
                         num_workers=conf['num_workers'],
                         pin_memory=True,
                         drop_last=True,
                         collate_fn=yolo_dataset_collate)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    if conf['Freeze_Train']:
        for param in model.backbone.parameters():
            param.requires_grad = False

    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                      epoch_step, epoch_step_val, gen, gen_val, end_epoch, CUDA)
        lr_scheduler.step()


def unfreeze_train(model, model_train, train_lines, val_lines, num_classes, num_train, num_val, yolo_loss,
                   loss_history):
    batch_size = conf['Unfreeze_batch_size']
    lr = eval(conf['Unfreeze_lr'])
    start_epoch = conf['Freeze_Epoch']
    end_epoch = conf['UnFreeze_Epoch']
    mosaic = conf['mosaic']
    optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)

    if conf['cosine_lr']:
        print('Using cosine_lr ')
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    train_dataset = YoloDataset(train_lines, conf['input_shape'], num_classes, train=True, mosaic=mosaic)
    val_dataset = YoloDataset(val_lines, conf['input_shape'], num_classes, train=False, mosaic=False)
    gen = DataLoader(train_dataset,
                     shuffle=True,
                     batch_size=batch_size,
                     num_workers=conf['num_workers'],
                     pin_memory=True,
                     drop_last=True,
                     collate_fn=yolo_dataset_collate)

    gen_val = DataLoader(val_dataset,
                         shuffle=True,
                         batch_size=batch_size,
                         num_workers=conf['num_workers'],
                         pin_memory=True,
                         drop_last=True,
                         collate_fn=yolo_dataset_collate)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    if conf['Freeze_Train']:
        for param in model.backbone.parameters():
            param.requires_grad = True

    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                      epoch_step, epoch_step_val, gen, gen_val, end_epoch, CUDA)
        lr_scheduler.step()


def train():
    obj = conf['object'][conf['obj_type']]
    class_names, num_classes = get_classes(obj['classes_path'])
    anchors, num_anchors = get_anchors(obj['anchors_path'])

    print(f'classes names:{class_names}')
    print(f'anchors:{anchors}')
    print(f'num workers:{conf["num_workers"]}')
    print(f'SPP args:{conf["spp"]}')
    print(f'Mosaic:{conf["mosaic"]}')
    print(f'Activation:{conf["activation"]}')
    print(f'Cosine LR:{conf["cosine_lr"]}')
    print(f'Label smoothing:{conf["label_smoothing"]}')

    print(f'Freeze_Train:{conf["Freeze_Train"]}')
    if conf['Freeze_Train']:
        print(f'Freeze epoch:{conf["Freeze_Epoch"]}')
        print(f'Freeze batch size:{conf["freeze_batch_size"]}')
        print(f'Freeze_lr epoch:{conf["freeze_lr"]}')

    print(f'Unfreeze epoch:{conf["UnFreeze_Epoch"]}')
    print(f'Unfreeze batch size:{conf["Unfreeze_batch_size"]}')
    print(f'Unfreeze_lr epoch:{conf["Unfreeze_lr"]}')

    model = YOLO(conf['anchors_mask'], num_classes, conf['activation'])
    if conf['spp']:
        model = YOLOSPP(conf['anchors_mask'], num_classes, conf['spp'], conf['activation'])

    weights_init(model)
    print('YOLOV3 Weights Init Done.')

    # 载入yolo weight
    if obj['model_path'] != '':
        print(f'loading weights:{obj["model_path"]}')

        load_weights(model, obj['model_path'])
        # pretrained_dict = torch.load(obj['model_path'], map_location=device)
        # model.load_state_dict(pretrained_dict, strict=False)
        # _t = model.state_dict()
        print('loading weights done.')

    model_train = model.train()

    if CUDA:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss = YOLOLoss(anchors,
                         num_classes,
                         conf['input_shape'],
                         CUDA,
                         conf['anchors_mask'],
                         label_smoothing=conf['label_smoothing'])

    loss_history = LossHistory("logs/")

    # load train val dataset txt
    with open(conf['train_annotation_path']) as f:
        train_lines = f.readlines()
    with open(conf['val_annotation_path']) as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    # 冻结训练
    if conf['Freeze_Train']:
        freeze_train(model, model_train, train_lines, val_lines, num_classes, num_train, num_val, yolo_loss,
                     loss_history)
    # 全网络训练
    unfreeze_train(model, model_train, train_lines, val_lines, num_classes, num_train, num_val, yolo_loss,
                   loss_history)


def main():
    train()


if __name__ == '__main__':
    main()
