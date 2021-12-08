import os
import random
import xml.etree.ElementTree as ET

from utils.utils import get_classes, load_yaml_conf

conf = load_yaml_conf('train.yaml')
obj = conf['object'][conf['obj_type']]

VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]

classes, classes_len = get_classes(obj['classes_path'])


def get_annotation_data(root: str, year: str, image_name: str, list_file):
    xml_file = open(os.path.join(root, f'VOCdevkit/VOC{year}/Annotations/{image_name}.xml'), encoding='utf-8')
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text

        # 排除不在类别列表中的数据和困难数据
        name = obj.find('name').text
        if name not in classes or int(difficult) == 1:
            continue

        cls_id = classes.index(name)
        xmlbox = obj.find('bndbox')
        box = (int(float(xmlbox.find('xmin').text)),
               int(float(xmlbox.find('ymin').text)),
               int(float(xmlbox.find('xmax').text)),
               int(float(xmlbox.find('ymax').text)))
        # xmin,ymin,xmax,ymax,classes_id
        list_file.write(' ' + ','.join([str(a) for a in box]) + ',' + str(cls_id))


def generate_train_val_test():
    random.seed(0)
    print('Generate train tainval val test txt in ImageSets/Main.')
    xml_file_path = os.path.join(obj['dataset_root'], 'VOCdevkit/VOC2007/Annotations')
    save_file_path = os.path.join(obj['dataset_root'], 'VOCdevkit/VOC2007/ImageSets/Main')
    xml_data = os.listdir(xml_file_path)
    total_xml = []
    for xml in xml_data:
        if xml.endswith('.xml'):
            total_xml.append(xml)

    num_xml = len(total_xml)
    list = range(num_xml)
    tv = int(num_xml * conf['trainval_percent'])
    tr = int(tv * conf['train_percent'])
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("train size", tr)
    ftrainval = open(os.path.join(save_file_path, 'trainval.txt'), 'w')
    ftest = open(os.path.join(save_file_path, 'test.txt'), 'w')
    ftrain = open(os.path.join(save_file_path, 'train.txt'), 'w')
    fval = open(os.path.join(save_file_path, 'val.txt'), 'w')

    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate train tainval val test txt in ImageSets/Main done.")


def generate_yolo_train_val():
    print("Generate 2007_train.txt and 2007_val.txt for train.")
    for year, image_set in VOCdevkit_sets:
        image_names = open(os.path.join(obj["dataset_root"], f'VOCdevkit/VOC{year}/ImageSets/Main/{image_set}.txt'),
                           encoding='utf-8').read().strip().split()
        # 2007_tarin.txt
        # 2007_val.txt
        list_file = open(f'{year}_{image_set}.txt', 'w', encoding='utf-8')
        for image_name in image_names:
            # write:xxx/xxx/xxx/jpg
            list_file.write(f'{os.path.abspath(obj["dataset_root"])}/VOCdevkit/VOC{year}/JPEGImages/{image_name}.jpg')
            # write: xmin ymin xmax ymax classed_id
            get_annotation_data(obj['dataset_root'], year, image_name, list_file)
            list_file.write('\n')

        list_file.close()
    print("Generate 2007_train.txt and 2007_val.txt for train done.")


def maia():
    generate_train_val_test()
    generate_yolo_train_val()


if __name__ == '__main__':
    maia()
