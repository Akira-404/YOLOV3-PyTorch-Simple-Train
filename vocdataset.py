import os
from random import sample, shuffle
import xml.etree.ElementTree as ET

import cv2
import torch
import numpy as np
from loguru import logger
from torchvision import transforms
from torch.utils.data.dataset import Dataset

from utils.utils import load_yaml


def rand(a: float = 0.0, b: float = 1.0) -> np.ndarray:
    return np.random.rand() * (b - a) + a


class VOCDataset(Dataset):

    # 初始化类
    def __init__(self, config: str):
        cwd = os.path.dirname(__file__)
        self.config = load_yaml(os.path.join(cwd, config))

        self._root = self.config['dataset_root']
        self._use_difficult = self.config['use_difficult']
        self._use_text = self.config['use_text']
        self._use_mosaic = self.config['use_mosaic']
        # self._mean = self.config['mean']
        # self._std = self.config['std']
        self._classes_path = os.path.join(cwd, self.config['classes_path'])
        self._anno_path = os.path.join(self._root, "Annotations", "{}.xml")
        self._img_path = os.path.join(self._root, "JPEGImages", "{}.jpg")
        self._imgset_path = os.path.join(self._root, "ImageSets", "Main", "{}.txt")

        logger.info(f'[VOCDataset]::dataset _root: {self._root}')
        logger.info(f'[VOCDataset]::use difficult: {self._use_difficult}')
        logger.info(f'[VOCDataset]::use text file: {self._use_text}')
        logger.info(f'[VOCDataset]::classes path: {self._classes_path}')
        logger.info(f'[VOCDataset]::annotation path: {self._anno_path}')
        logger.info(f'[VOCDataset]::image path: {self._img_path}')
        logger.info(f'[VOCDataset]::image set path: {self._imgset_path}')
        # logger.info(f'[VOCDataset]::mean: {self._mean}')
        # logger.info(f'[VOCDataset]::std: {self._std}')

        # 读取trainval.txt中内容
        with open(self._imgset_path.format(self._use_text)) as f:
            self.img_ids = f.readlines()

        # ['000009', '000052']
        self.img_ids = [x.strip() for x in self.img_ids]

        # load classes yaml
        self.classes_cfg = load_yaml(self._classes_path)
        self.classes_names = self.classes_cfg['names']
        self.name2id = dict(zip(self.classes_names, range(len(self.classes_names))))
        logger.info(f'[VOCDataset]::[name:id]: {self.name2id}')

        logger.info("Dataset init finished  ! !")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        index = index % len(self.img_ids)
        img_id = self.img_ids[index]
        img_path = self._img_path.format(img_id)
        anno_path = self._anno_path.format(img_id)

        # TODO:mosaic function
        # if self._use_mosaic:
        #     if rand() < 0.5:
        #         image, box = get_random_data_mosaic()
        #     else:
        #         image, box = get_random_data()
        # else:
        #     image, box = get_random_data()

        image, boxes, classes = get_random_data(img_path, anno_path, self.name2id, self._use_difficult)

        return image, boxes, classes


def get_random_data(img_path: str,
                    anno_path: str,
                    name2id: dict,
                    use_difficult: bool,
                    resize: tuple,
                    jitter: float = .3,
                    hue: float = .1,
                    sat: float = 1.5,
                    val: float = 1.5,
                    random: bool = True):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    anno = ET.parse(anno_path).getroot()  # 读取xml文档的根节点
    boxes = []
    classes = []

    for obj in anno.iter("object"):
        difficult = int(obj.find("difficult").text) == 1
        if not use_difficult and difficult:
            continue

        bndbox = obj.find("bndbox")
        box = [
            bndbox.find("xmin").text,
            bndbox.find("ymin").text,
            bndbox.find("xmax").text,
            bndbox.find("ymax").text,
        ]
        TO_REMOVE = 1  # 由于像素是网格存储，坐标2实质表示第一个像素格，所以-1
        box = tuple(
            map(lambda x: x - TO_REMOVE, list(map(float, box)))
        )
        boxes.append(box)

        name = obj.find("name").text.lower().strip()
        classes.append(name2id[name])  # 将类别映射回去

    boxes = np.array(boxes, dtype=np.float32)

    # image resize
    ih, iw, ic = img.shape
    h, w = resize
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)

    img = cv2.resize(img, resize, interpolation=cv2.INTER_CUBIC)

    # TODO:
    # image add gray bar
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))

    # flip image
    flip = rand() < .5
    img = img if flip else cv2.flip(img, 1)

    # RGB->HSV->RGB
    image_data = np.array(img, np.uint8)

    r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    #   将图像转到HSV上
    hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
    dtype = image_data.dtype
    #   应用变换
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    img = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

    # change bbox
    if len(boxes) > 0:
        np.random.shuffle(boxes)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / iw + dx
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / ih + dy
        if flip: boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > w] = w
        boxes[:, 3][boxes[:, 3] > h] = h
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]
    return img, boxes, classes


def get_random_data_mosaic():
    img = None,
    box = None
    return img, box


if __name__ == '__main__':
    dataset = VOCDataset('data/voc/config.yaml')  # 实例化一个对象
    image, boxes, classes = dataset[0]  # 返回第一张图像及box和对应的类别
    logger.info(image.shape)
    logger.info(boxes)
    logger.info(classes)

    # 这里简单做一下可视化
    # 由于opencv读入是矩阵，而img现在是tensor，因此，首先将tensor转成numpy.array
    img_ = (image.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)  # 注意由于图像像素分布0-255，所以转成uint8
    logger.info(img_.shape)
    cv2.imshow('test', img_)
    cv2.waitKey(0)
