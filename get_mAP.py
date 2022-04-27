import os
import yaml
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from main import Predict
from utils.utils import get_classes, load_yaml_conf
from utils.utils_map import get_coco_map, get_map

_map_out_path = 'map_out'
_conf = load_yaml_conf('predict.yaml')
_type = _conf['object'][_conf['obj_type']]
image_ids = open(
    os.path.join(_type['dataset_root'], "VOCdevkit/VOC2007/ImageSets/Main/test.txt")).read().strip().split()

if not os.path.exists(_map_out_path):
    os.makedirs(_map_out_path)
if not os.path.exists(os.path.join(_map_out_path, 'ground-truth')):
    os.makedirs(os.path.join(_map_out_path, 'ground-truth'))
if not os.path.exists(os.path.join(_map_out_path, 'detection-results')):
    os.makedirs(os.path.join(_map_out_path, 'detection-results'))
if not os.path.exists(os.path.join(_map_out_path, 'images-optional')):
    os.makedirs(os.path.join(_map_out_path, 'images-optional'))

class_names, _ = get_classes(_type['classes_path'])
print('Load model.')
predict = Predict('predict.yaml')
print('Load model done.')

print('Get predict result.')
for image_id in tqdm(image_ids):
    image_path = os.path.join(_type['dataset_root'], "JPEGImages/" + image_id + ".jpg")
    image = Image.open(image_path)
    predict.get_map_txt(image_id, image, class_names, _map_out_path)
print('Get predict result done.')

print("Get ground truth result.")
for image_id in tqdm(image_ids):
    with open(os.path.join(_map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
        root = ET.parse(
            os.path.join(_type['dataset_root'], "Annotations/" + image_id + ".xml")).getroot()
        for obj in root.findall('object'):
            difficult_flag = False
            if obj.find('difficult') is not None:
                difficult = obj.find('difficult').text
                if int(difficult) == 1:
                    difficult_flag = True
            obj_name = obj.find('name').text
            if obj_name not in class_names:
                continue
            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text

            if difficult_flag:
                new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
            else:
                new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
print("Get ground truth result done.")

print("Get map.")
get_map(_conf['minoverlap'], True, path=_map_out_path)
print("Get map done.")

print("Get map.")
get_coco_map(class_names=class_names, path=_map_out_path)
print("Get map done.")
