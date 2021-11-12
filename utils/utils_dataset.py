import os
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser('YOLO TO VOC')
parser.add_argument('-i', '--images_path', type=str,
                    default='/home/cv/AI_Data/head_datas_yolo/VOCdevkit/VOC2007/JPEGImages')
parser.add_argument('-l', '--labels_path', type=str,
                    default='/home/cv/AI_Data/head_datas_yolo/labels')
parser.add_argument('-a', '--annotations_path', type=str,
                    default='/home/cv/AI_Data/head_datas_yolo/VOCdevkit/VOC2007/Annotations')
parser.add_argument('-c', '--classes_path', type=str, default='/home/cv/AI_Data/head_datas_yolo/classes.txt')
args = parser.parse_args()

# 类别
classes = ["head"]

# 图片的高度、宽度、深度
img_h = img_w = img_d = 0


def write_xml(imgname, img_w, img_h, img_d, filepath, labeldicts):
    '''
    imgname: 没有扩展名的图片名称
    '''

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


labels = os.listdir(args.labels_path)
for label in tqdm(labels):
    with open(os.path.join(args.labels_path, label), 'r') as f:
        img_id = os.path.splitext(label)[0]
        contents = f.readlines()
        labeldicts = []
        for content in contents:
            img = np.array(Image.open(os.path.join(args.images_path, label).replace('txt', 'jpg')))

            # 图片的高度和宽度
            img_h, img_w, img_d = img.shape[0], img.shape[1], img.shape[2]
            content = content.strip('\n').split()
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
        write_xml(img_id, img_w, img_h, img_d, os.path.join(args.annotations_path, label).replace('txt', 'xml'),
                  labeldicts)
