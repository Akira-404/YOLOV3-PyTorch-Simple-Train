import os
import copy
from random import sample, shuffle
import xml.etree.ElementTree as ET

import cv2
import torch
import numpy as np
from loguru import logger
from torchvision import transforms
from torch.utils.data.dataset import Dataset

from utils.common import load_yaml, read_txt
from utils.image import image_normalization
from utils.data_augmentation import *


class VOCDataset(Dataset):

    # 初始化类
    def __init__(self, config: str, train: bool = True):
        cwd = os.path.dirname(__file__)
        self.config = load_yaml(os.path.join(cwd, config))
        self.istrain = train

        self._root = self.config['train_dataset_root'] if train else self.config['test_dataset_root']

        self._use_difficult = self.config['use_difficult']
        self._use_text = self.config['use_text']
        self._use_mosaic = self.config['use_mosaic']
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

        # 读取trainval.txt中内容
        self.img_ids = read_txt(self._imgset_path.format(self._use_text))

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

        image, boxes = get_random_data(img_path,
                                       anno_path,
                                       self.name2id,
                                       self._use_difficult,
                                       tuple(self.config['image_shape']),
                                       random=self.istrain)

        image = image_normalization(np.array(image, dtype=np.float32))
        image = np.transpose(image, (2, 0, 1))  # (h,w,c)->(c,h,w)

        if len(boxes) != 0:
            # This part is from darknet yolov3
            # darknete box=[xmin,xmax,ymin,ymax]
            # dw = 1. / (size[0])
            # dh = 1. / (size[1])
            # x = (box[0] + box[1]) / 2.0 - 1
            # y = (box[2] + box[3]) / 2.0 - 1
            # w = box[1] - box[0]
            # h = box[3] - box[2]
            # x = x * dw
            # w = w * dw
            # y = y * dh
            # h = h * dh

            boxes_tmp = copy.deepcopy(boxes)
            # this box =[xmin,ymin,xmax,ymax]
            # [cx,cy.]=([x1,y1]+[x2,y2])/2
            boxes[:, [0, 1]] = (boxes_tmp[:, [0, 1]] + boxes_tmp[:, [2, 3]]) / 2 - 1
            # [w,h]=[x2,y2]-[x1,y1]
            boxes[:, [2, 3]] = boxes_tmp[:, [2, 3]] - boxes_tmp[:, [0, 1]]

            # [cx,w]/iw
            # [cy,h]/ih
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.config['image_shape'][1]  # w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.config['image_shape'][0]  # h
            # return box=[cx,cy,w,h]
        return image, boxes


def get_random_data(img_path: str = None,
                    anno_path: str = None,
                    name2id: dict = None,
                    use_difficult: bool = False,
                    resize: tuple = None,
                    hue: float = .1,
                    sat: float = .7,
                    val: float = .4,
                    random: bool = False):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    anno = ET.parse(anno_path).getroot()  # 读取xml文档的根节点
    boxes = []

    for obj in anno.iter("object"):
        difficult = int(obj.find("difficult").text) == 1
        if not use_difficult and difficult:
            continue

        # <<< this part is use to train wider face dataset <<<
        if int(obj.find("truncated").text) == 1:
            continue

        if int(obj.find("difficult").text) == 1:
            continue

        if int(obj.find("blur").text) == 2:
            continue

        # if int(obj.find("expression").text) == 2:
        #     continue

        if int(obj.find("illumination").text) == 1:
            continue

        if int(obj.find("invalid").text) == 1:
            continue

        # if int(obj.find("occlusion").text) == 1:
        #     continue

        # if int(obj.find("pose").text) == 1:
        #     continue
        # <<< this part is use to train wider face dataset <<<

        bndbox = obj.find("bndbox")
        name = obj.find("name").text.lower().strip()
        # classes.append(name2id[name])  # 将类别映射回去
        box = [
            bndbox.find("xmin").text,
            bndbox.find("ymin").text,
            bndbox.find("xmax").text,
            bndbox.find("ymax").text,
            name2id[name]
        ]

        boxes.append(box)

    boxes = np.array(boxes, dtype=np.float32)

    # data augmentation
    # train:random=True
    # val & test :random=False

    # img resize
    # img_resize = image_resize_letterbox(img, resize) if random else img
    img_resize = letterbox(img, resize) if random else img

    # image flip
    flip = rand() < .5 if random else False
    img_resize = image_flip(img_resize) if flip else img_resize

    # change bbox
    boxes = label_adjust(img.shape[:2], resize, boxes, flip) if random else boxes

    # RGB->HSV->RGB
    img_resize = color_jittering(img_resize, hue, sat, val)

    return img_resize, boxes


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes


if __name__ == '__main__':
    dataset = VOCDataset('data/voc/config.yaml')  # 实例化一个对象
    image, boxes = dataset[0]  # 返回第一张图像及box和对应的类别
    logger.info(image.shape)
    logger.info(boxes)

    img = (image * 255).astype(np.uint8).transpose(1, 2, 0)  # 注意由于图像像素分布0-255，所以转成uint8
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # [cx,w]*iw
    # [cy,h]*ih
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * img.shape[1]
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * img.shape[0]

    # [h,w]=[h/2,w/2]
    boxes[:, 2:4] = boxes[:, 2:4] / 2
    tmp = copy.deepcopy(boxes)

    # [x1,,y1]=[cx-w/2,cy-h/2]
    boxes[:, 0:2] = tmp[:, 0:2] - tmp[:, 2:4]
    # [x2,,y2]=[cx+w/2,cy+h/2]
    boxes[:, 2: 4] = tmp[:, 0:2] + tmp[:, 2:4]

    for box in boxes:
        x1, y1 = int(box[0]), int(box[1])
        x2, y2 = int(box[2]), int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1, 1)
    cv2.imshow('test', img)
    cv2.waitKey(0)
