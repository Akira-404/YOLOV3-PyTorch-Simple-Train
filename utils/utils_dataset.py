import os
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

parser = argparse.ArgumentParser('YOLO TO VOC')
parser.add_argument('-r', '--root', type=str, default='/home/cv/AI_Data/widerperson',
                    help='yolo dataset root')
args = parser.parse_args()

assert os.path.exists(args.root) is True, f'{args.root} is error'

_anno_path = os.path.join(args.root, 'VOCdevkit/VOC2007/Annotations')
_jpeg_path = os.path.join(args.root, 'VOCdevkit/VOC2007/JPEGImages')
if os.path.exists(_anno_path) is False:
    os.makedirs(_anno_path)
if os.path.exists(_jpeg_path) is False:
    os.makedirs(_jpeg_path)

images_path = os.path.join(args.root, 'images')
labels_path = os.path.join(args.root, 'labels')
classes_path = os.path.join(args.root, 'VOCdevkit', 'VOC2007', 'classes.txt')

# 获取类别
with open(classes_path, 'r') as f:
    cls = f.read().split()
classes = cls
print(f'classes:{classes}')

# 图片的高度、宽度、深度
img_h = img_w = img_d = 0


def write_xml(imgname: str, img_w: int, img_h: int, img_d: int, filepath: str, labeldicts: list):
    # 创建Annotation根节点
    root = ET.Element('annotation')

    # 创建filename子节点，无扩展名                 
    ET.SubElement(root, 'filename').text = str(imgname)

    # 创建size子节点 
    sizes = ET.SubElement(root, 'size')
    ET.SubElement(sizes, 'width').text = str(img_w)
    ET.SubElement(sizes, 'height').text = str(img_h)
    ET.SubElement(sizes, 'depth').text = str(img_d)

    for labeldict in labeldicts:
        objects = ET.SubElement(root, 'object')
        ET.SubElement(objects, 'name').text = labeldict['name']
        ET.SubElement(objects, 'pose').text = 'Unspecified'
        ET.SubElement(objects, 'truncated').text = '0'
        ET.SubElement(objects, 'difficult').text = '0'
        bndbox = ET.SubElement(objects, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(labeldict['xmin']))
        ET.SubElement(bndbox, 'ymin').text = str(int(labeldict['ymin']))
        ET.SubElement(bndbox, 'xmax').text = str(int(labeldict['xmax']))
        ET.SubElement(bndbox, 'ymax').text = str(int(labeldict['ymax']))
    tree = ET.ElementTree(root)
    tree.write(filepath, encoding='utf-8')


def yolo2voc():
    for label in tqdm(os.listdir(labels_path)):
        label_path = os.path.join(labels_path, label)
        # print(label_path)
        with open(label_path, 'r') as f:
            # 获取图片名
            img_name = os.path.splitext(label)[0]

            # 读取图片
            img_path = os.path.join(images_path, label).replace('txt', 'jpg')
            if not os.path.exists(img_path):
                # print(img_path)
                continue
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = np.array(Image.open(os.path.join(images_path, label).replace('txt', 'jpg')))

            # 读取lable内容
            contents = f.readlines()
            labeldicts = []
            for content in contents:

                # 图片的高度和宽度
                # print(img.shape)
                img_h, img_w, img_d = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
                content = content.strip('\n').split()
                if len(content)<5:
                    continue
                x = float(content[1]) * img_w
                y = float(content[2]) * img_h
                w = float(content[3]) * img_w
                h = float(content[4]) * img_h

                # 坐标的转换，x_center y_center width height -> xmin ymin xmax ymax
                new_dict = {'name': classes[int(content[0])],
                            'difficult': '0',
                            'xmin': x + 1 - w / 2,
                            'ymin': y + 1 - h / 2,
                            'xmax': x + 1 + w / 2,
                            'ymax': y + 1 + h / 2
                            }
                labeldicts.append(new_dict)

            write_xml(img_name, img_w, img_h, img_d, os.path.join(_anno_path, label).replace('txt', 'xml'),
                      labeldicts)


def check_dataset():
    annos = os.listdir(_anno_path)
    images = os.listdir(_jpeg_path)
    print(len(annos))
    print(len(images))
    assert len(annos) == len(images), f'anns len!=image len'

    for anno in annos:
        anno = os.path.join(_anno_path, anno)
        image = anno.replace('Annotations', 'JPEGImages').replace('xml', 'jpg')
        assert os.path.exists(image) is True, f'{image} is error'


if __name__ == '__main__':
    yolo2voc()
    # check_dataset()
