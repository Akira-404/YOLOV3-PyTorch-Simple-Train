import os
import numpy as np
from PIL import Image
import yaml
import torch


# 加载yaml配置文件
def load_yaml_conf(conf_path: str) -> dict:
    """
    :param conf_path: xxx/xxx/xxx.yaml
    :return: dict
    """
    assert conf_path.endswith(".yaml") is True, f'file type must be yaml'
    assert os.path.exists(conf_path) is True, f'{conf_path}is error'
    if conf_path.endswith(".yaml"):
        with open(conf_path, 'r') as f:
            conf = yaml.safe_load(f)
    return conf


# 加载权重
def load_weights(model, model_path: str, device: str):
    """
    :param model: net model
    :param model_path: xxx/xxx/xxx.pth
    :param device: cpu or gpu
    :return: None
    """
    print(f'Load weights {model_path}')
    model_dict = model.state_dict()
    _model_dict = {}
    pretrained_dict = torch.load(model_path, map_location=device)

    for k, v in model_dict.items():
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


def img2rgb(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size: tuple, letterbox_image: bool = False):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def get_classes(classes_path: str):
    """
    :param classes_path: support file type:yaml or txt
    :return: classes names and length of classes names
    """

    if classes_path.endswith(".yaml"):
        with open(classes_path, 'r') as f:
            data = yaml.safe_load(f)
        class_names = data['names']

    elif classes_path.endswith(".txt"):
        with open(classes_path, 'r', encoding='utf-8') as f:
            class_names = f.readlines()

    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def get_anchors(anchors_path: str):
    """
    :param anchors_path: support file type:yaml or txt
    :return: anchors and length of classes names
    """
    anchors = []
    if anchors_path.endswith(".yaml"):
        with open(anchors_path, 'r') as f:
            data = yaml.safe_load(f)
        anchors = data['anchors']

    elif anchors_path.endswith(".txt"):
        with open(anchors_path, 'r', encoding='utf-8') as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]

    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def preprocess_input(image):
    image /= 255.0
    return image


def check_dataset_classes(dataset_root: str, save_path: str = '../model_data/my_classes.yaml'):
    ...
