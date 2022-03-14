import os
# import scipy.io as sio
import shutil
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import cv2
from tqdm import tqdm


def make_voc_dir(voc_root: str):
    anno = os.path.join(voc_root, 'Annotations')
    # labels 目录若不存在，创建labels目录。若存在，则清空目录
    if not os.path.exists(anno):
        os.makedirs(anno)

    images_sets = os.path.join(voc_root, 'ImageSets')
    if not os.path.exists(images_sets):
        os.makedirs(images_sets)
        os.makedirs(os.path.join(images_sets, 'Main'))

    jpeg_images = os.path.join(voc_root, 'JPEGImages')
    if not os.path.exists(jpeg_images):
        os.makedirs(jpeg_images)


def toVOC(root: str):
    classes = {'1': 'pedestrians',  # 行人
               '2': 'riders',  # 骑车的人
               '3': 'partially',  # 被挡住了一部分的人
               '4': 'ignore',  # 假人
               '5': 'crowd'}  # 拥挤的人群

    make_voc_dir(root)

    data_file = ['train.txt', 'val.txt']
    print(f'Transform the dataset.')
    for file in data_file:
        print(f'Current file:{file}')
        txt_file = os.path.join(root, file)

        with open(txt_file, 'r') as f:
            img_idx = [x for x in f.read().splitlines()]

        for imgId in tqdm(img_idx):
            objCount = 0  # 一个标志位，用来判断该img是否包含我们需要的标注
            filename = imgId + '.jpg'  # 000001.jpg
            img_path = os.path.join(root, 'Images', filename)  # xxx/xxx/VOCdevkit/VOC2007/WiderPerson/Images/000001.jpg

            # print(f'Img :{img_path}')
            assert os.path.exists(img_path), 'img is not found'

            img = cv2.imread(img_path)

            width = img.shape[1]  # 获取图片尺寸
            height = img.shape[0]  # 获取图片尺寸

            node_root = Element('annotation')

            node_folder = SubElement(node_root, 'folder')
            node_folder.text = 'JPEGImages'

            node_filename = SubElement(node_root, 'filename')
            # node_filename.text = 'VOC2007/JPEGImages/%s' % filename
            node_filename.text = 'WiderPerson/%s' % filename

            node_size = SubElement(node_root, 'size')

            node_width = SubElement(node_size, 'width')
            node_width.text = '%s' % width

            node_height = SubElement(node_size, 'height')
            node_height.text = '%s' % height

            node_depth = SubElement(node_size, 'depth')
            node_depth.text = '3'

            label_path = img_path.replace('Images', 'Annotations') + '.txt'
            with open(label_path) as file:
                line = file.readline()
                count = int(line.split('\n')[0])  # 里面行人个数
                line = file.readline()
                while line:
                    cls_id = line.split(' ')[0]
                    xmin = int(line.split(' ')[1]) + 1
                    ymin = int(line.split(' ')[2]) + 1
                    xmax = int(line.split(' ')[3]) + 1
                    ymax = int(line.split(' ')[4].split('\n')[0]) + 1
                    line = file.readline()

                    cls_name = classes[cls_id]

                    obj_width = xmax - xmin
                    obj_height = ymax - ymin

                    difficult = 0
                    if obj_height <= 6 or obj_width <= 6:
                        difficult = 1

                    node_object = SubElement(node_root, 'object')
                    node_name = SubElement(node_object, 'name')
                    node_name.text = cls_name
                    node_difficult = SubElement(node_object, 'difficult')
                    node_difficult.text = '%s' % difficult
                    node_bndbox = SubElement(node_object, 'bndbox')
                    node_xmin = SubElement(node_bndbox, 'xmin')
                    node_xmin.text = '%s' % xmin
                    node_ymin = SubElement(node_bndbox, 'ymin')
                    node_ymin.text = '%s' % ymin
                    node_xmax = SubElement(node_bndbox, 'xmax')
                    node_xmax.text = '%s' % xmax
                    node_ymax = SubElement(node_bndbox, 'ymax')
                    node_ymax.text = '%s' % ymax
                    node_name = SubElement(node_object, 'pose')
                    node_name.text = 'Unspecified'
                    node_name = SubElement(node_object, 'truncated')
                    node_name.text = '0'

            xml = tostring(node_root, pretty_print=True)  # 'annotation'
            dom = parseString(xml)
            xml_name = filename.replace('.jpg', '.xml')
            xml_path = os.path.join(root, 'Annotations', xml_name)

            if os.path.exists(xml_path):
                # print(f'Already exists:{xml_path}')
                continue
            with open(xml_path, 'wb') as f:
                f.write(xml)

            shutil.copy(img_path, os.path.join(root, 'JPEGImages', filename))


def display(root: str):
    train_txt = os.path.join(root, 'train.txt')
    with open(train_txt, 'r') as f:
        img_ids = [x for x in f.read().splitlines()]
    print(img_ids)

    for img_id in img_ids:
        img_path = os.path.join(root, 'Images', img_id) + '.jpg'
        print(img_path)

        img = cv2.imread(img_path)

        im_h = img.shape[0]
        im_w = img.shape[1]

        label_path = img_path.replace('Images', 'Annotations') + '.txt'

        with open(label_path) as file:
            line = file.readline()
            count = int(line.split('\n')[0])  # 里面行人个数
            line = file.readline()
            while line:
                cls = int(line.split(' ')[0])
                # < class_label =1: pedestrians > 行人
                # < class_label =2: riders >      骑车的
                # < class_label =3: partially-visible persons > 遮挡的部分行人
                # < class_label =4: ignore regions > 一些假人，比如图画上的人
                # < class_label =5: crowd > 拥挤人群，直接大框覆盖了
                if cls == 1 or cls == 2 or cls == 3:
                    xmin = float(line.split(' ')[1])
                    ymin = float(line.split(' ')[2])
                    xmax = float(line.split(' ')[3])
                    ymax = float(line.split(' ')[4].split('\n')[0])
                    img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                line = file.readline()
        cv2.imshow('result', img)
        cv2.waitKey(0)


def display_voc(voc_root: str):
    try:
        import xml.etree.cElementTree as ET
    except ImportError:
        import xml.etree.ElementTree as ET

    train_txt = os.path.join(voc_root, 'train.txt')
    with open(train_txt, 'r') as f:
        img_ids = [x for x in f.read().splitlines()]

    for i, img_id in enumerate(img_ids):
        img_path = os.path.join(voc_root, 'JPEGImages', img_id) + '.jpg'
        print(i, img_path)

        img = cv2.imread(img_path)
        im_h = img.shape[0]
        im_w = img.shape[1]
        label_path = img_path.replace('JPEGImages', 'Annotations').replace('jpg', 'xml')

        tree = ET.ElementTree(file=label_path)
        root = tree.getroot()
        ObjectSet = root.findall('object')
        ObjBndBoxSet = {}
        for Object in ObjectSet:
            ObjName = Object.find('name').text
            # print(f'Object name:{ObjName}')

            BndBox = Object.find('bndbox')
            x1 = int(BndBox.find('xmin').text)
            y1 = int(BndBox.find('ymin').text)
            x2 = int(BndBox.find('xmax').text)
            y2 = int(BndBox.find('ymax').text)
            # print(f'x1"{x1},y1:{y1},x2:{x2},y2:{y2}')


def match_txt_img(root: str):
    data_file = ['train.txt', 'val.txt']

    for file in data_file:
        print(f'Match txt and img current file:{file}')
        file_path = os.path.join(root, file)
        with open(file_path, 'r') as f:
            img_ids = [x for x in f.read().splitlines()]

        for img_id in tqdm(img_ids):
            img_path = os.path.join(root, 'Images', img_id) + '.jpg'
            # print(f'file:{file}')
            # print(f'image path:{img_path}')
            assert os.path.exists(img_path), 'img is not found'

            img = cv2.imread(img_path)
            im_h = img.shape[0]
            im_w = img.shape[1]
            label_path = img_path.replace('Images', 'Annotations') + '.txt'
            # print(f'label path:{label_path}')
            assert os.path.exists(label_path), 'label is not found'
    print(f'All txt files and pictures in the training set and validation set have been matched')


def match_xml_img(root: str):
    data_file = ['train.txt', 'val.txt']
    # data_file = [ 'val.txt']

    for file in data_file:
        print(f'Match xml and img current file:{file}')
        file_path = os.path.join(root, file)
        with open(file_path, 'r') as f:
            img_ids = [x for x in f.read().splitlines()]

        for img_id in tqdm(img_ids):
            img_path = os.path.join(root, 'JPEGImages', img_id) + '.jpg'
            # print(f'file:{file}')
            # print(f'image path:{img_path}')
            assert os.path.exists(img_path), f'img:{img_path} is not found'

            img = cv2.imread(img_path)
            im_h = img.shape[0]
            im_w = img.shape[1]
            label_path = img_path.replace('JPEGImages', 'Annotations').replace('jpg', 'xml')
            # print(f'label path:{label_path}')
            assert os.path.exists(label_path), f'label{label_path} is not found'

    print(f'All xml files and pictures in the training set and validation set have been matched')


if __name__ == '__main__':
    root='/home/ubuntu/data/VOCdevkit/VOC2007/WiderPerson'
    match_txt_img(root)
    toVOC(root)

    # display('/Users/lee/Desktop/dataset/Person/VOCdevkit/VOC2007/WiderPerson')
    # display_voc('/Users/lee/Desktop/dataset/Person/VOCdevkit/VOC2007/WiderPerson')

    match_xml_img(root)
