import os
import numpy as np
from PIL import Image
import yaml
import torch
from tqdm import tqdm
import xml.etree.ElementTree as ET
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# 加载yaml配置文件
def load_yaml_conf(conf_path: str) -> dict:
    """
    :param conf_path: xxx/xxx/xxx.yaml
    :return: dict
    """
    assert conf_path.endswith(".yaml") is True, f'file type must be yaml'
    assert os.path.exists(conf_path) is True, f'{conf_path}is error'
    if conf_path.endswith(".yaml"):
        with open(conf_path, 'r', encoding='UTF-8') as f:
            conf = yaml.safe_load(f)
    return conf


# 加载权重
def load_weights(model, model_path: str, device, ignore_track: bool = False):
    print(f'Load weights {model_path}')
    model_dict = model.state_dict()
    _model_dict = {}
    pretrained_dict = torch.load(model_path, map_location=device)

    for k, v in model_dict.items():

        # pytorch 0.4.0后BN layer新增 num_batches_tracked 参数
        # ignore_track=False:加载net中的 num_batches_tracked参数
        # ignore_track=True:忽略加载net中的 num_batches_tracked参数
        if 'num_batches_tracked' in k and ignore_track:
            print('pass->', k)
        else:
            _model_dict[k] = v
    cnt = 0
    load_dict = {}
    pretrained_dict = pretrained_dict['model'] if 'model' in pretrained_dict.keys() else pretrained_dict

    for kv1, kv2 in zip(_model_dict.items(), pretrained_dict.items()):
        if np.shape(kv1[1]) == np.shape(kv2[1]):
            load_dict[kv1[0]] = kv2[1]
            cnt += 1

    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    print(f'loaded:{cnt}/{len(pretrained_dict)}')


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


# def preprocess_input(image):
#     image /= 255.0
#     return image


def read_xml(path: str):
    tree = ET.parse(path)
    return tree


def check_dataset(conf: dict):
    import matplotlib.pyplot as plt
    obj = conf['object'][conf['obj_type']]
    annoa_root = os.path.join(obj['dataset_root'], f'VOCdevkit/VOC{conf["year"]}/Annotations')

    assert os.path.exists(annoa_root) is True, f'{annoa_root} is error'
    # dataset_info = {'total': 0, 'difficult': 0}
    dataset_info = {}
    xmls = os.listdir(annoa_root)
    for xml in tqdm(xmls):
        # dataset_info['total'] += 1
        xml_p = os.path.join(annoa_root, xml)
        assert os.path.exists(xml_p) is True, f'{xml_p} is error'
        tree = read_xml(xml_p)
        root = tree.getroot()

        for obj in root.iter('object'):
            # if obj.find('difficult').text == '1':
            #     dataset_info['difficult'] += 1

            name = obj.find('name').text
            if name not in dataset_info.keys():
                dataset_info[name] = 0
            dataset_info[name] += 1
    print(dataset_info)

    import matplotlib.pyplot as plt
    fig = plt.figure('Dataset Info Pie Chart')

    # 柱状图
    data = dataset_info.values()
    classes = [f'{v[0]}:{v[1]}' for v in dataset_info.items()]

    # 添加图形对象
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('equal')
    ax.pie(data, labels=classes, autopct='%1.2f%%')
    plt.show()


# 检查预加载权重细节
def check_pretrained_weight(model_path: str, save_name: str, device: str):
    pretrained_dict = torch.load(model_path, map_location=device)
    print(f'model layer num:{len(pretrained_dict)}')

    if os.path.exists(f'{save_name}') is True:
        os.remove(f'{save_name}')

    f = open(f'{save_name}', 'a')
    for i, (k, v) in enumerate(pretrained_dict['model'].items()):
        f.write(str(k) + '\t' + str(np.shape(v)))
        f.write('\n')
    f.close()
    print(f'write:{save_name}')


def package_data(image_shape, top_label, top_conf, top_boxes, class_names):
    data = []
    for i, c in list(enumerate(top_label)):
        predicted_class = class_names[int(c)]
        box = top_boxes[i]
        score = top_conf[i]

        # top, left, bottom, right = box
        y0, x0, y1, x1 = box
        # x0, y0, x1, y1 = box

        y0 = max(0, np.floor(y0).astype('int32'))
        x0 = max(0, np.floor(x0).astype('int32'))
        y1 = min(image_shape[1], np.floor(y1).astype('int32'))
        x1 = min(image_shape[0], np.floor(x1).astype('int32'))
        # y1 = min(image.size[1], np.floor(y1).astype('int32'))
        # x1 = min(image.size[0], np.floor(x1).astype('int32'))

        # label = '{} {:.2f}'.format(predicted_class, score)
        # label = label.encode('utf-8')
        # print(label, x0, y0, x1, y1)
        item = {
            'label': predicted_class,
            'score': float(score),
            'height': int(y1 - y0),
            'left': int(x0),
            'top': int(y0),
            'width': int(x1 - x0)
        }
        data.append(item)
    return data


if __name__ == '__main__':
    conf = load_yaml_conf('../train.yaml')
    check_dataset(conf)
