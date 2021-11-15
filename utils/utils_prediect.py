import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo import YOLO
from utils.utils import (img2rgb, get_anchors, get_classes, preprocess_input, resize_image, load_yaml_conf)

from utils.utils_bbox import DecodeBox


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
    load_dict = {}
    pretrained_dict = pretrained_dict['model'] if pretrained_dict.keys in ['model'] else pretrained_dict
    for kv1, kv2 in zip(_model_dict.items(), pretrained_dict.items()):
        if np.shape(kv1[1]) == np.shape(kv2[1]):
            load_dict[kv1[0]] = kv2[1]
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    return model


class Predict:
    def __init__(self, conf_path: str):
        """
        :param conf_path: xxx.yaml
        """
        super(Predict, self).__init__()
        self.conf = load_yaml_conf(conf_path)

        self.CUDA = True if torch.cuda.is_available() and self.conf['cuda'] else False

        assert os.path.exists(self.conf['classes_path']) is True, f'{self.conf["classes_path"]} is error'
        assert os.path.exists(self.conf['anchors_path']) is True, f'{self.conf["anchors_path"]} is error'
        assert os.path.exists(self.conf['model_path']) is True, f'{self.conf["model_path"]} is error'

        self.class_names, self.num_classes = get_classes(self.conf['classes_path'])
        self.anchors, self.num_anchors = get_anchors(self.conf['anchors_path'])

        self.bbox_util = DecodeBox(self.anchors,
                                   self.num_classes,
                                   (self.conf['input_shape'][0], self.conf['input_shape'][1]),
                                   self.conf['anchors_mask'])

        #   画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate_model()

    def generate_model(self):
        self.net = YOLO(self.conf['anchors_mask'], self.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() and self.conf['cuda'] else 'cpu')

        self.net = load_weights(self.net, self.conf['model_path'], device, ignore_track=True)

        # self.net.load_state_dict(torch.load(self.conf['model_path'], map_location=device))

        self.net = self.net.eval()
        print(f'{self.conf["model_path"]} model,anchors,classes loaded')

        if self.CUDA:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])  # w,h
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image = img2rgb(image)
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        image_data = resize_image(image,
                                  (self.conf['input_shape'][1], self.conf["input_shape"][0]),
                                  self.conf['letterbox_image'])
        #   添加上batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.cuda() if self.CUDA else images

            #   将图像输入网络当中进行预测！
            outputs = self.net(images)
            # outputs shape: (3,batch_size,x,y,w,h,conf,classes)
            outputs = self.bbox_util.decode_box(outputs)
            # results=outputs
            #   将预测框进行堆叠，然后进行非极大抑制
            # results shape:(len(prediction),num_anchors,4)
            results = self.bbox_util.nms_(torch.cat(outputs, 1),
                                          self.num_classes,
                                          self.conf['input_shape'],
                                          image_shape,
                                          self.conf['letterbox_image'],
                                          conf_thres=self.conf['confidence'],
                                          nms_thres=self.conf['nms_iou'])

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        #   设置字体与边框厚度

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.conf['input_shape']), 1))

        #   图像绘制
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            # top, left, bottom, right = box
            y0, x0, y1, x1 = box
            # x0, y0, x1, y1 = box

            y0 = max(0, np.floor(y0).astype('int32'))
            x0 = max(0, np.floor(x0).astype('int32'))
            y1 = min(image.size[1], np.floor(y1).astype('int32'))
            x1 = min(image.size[0], np.floor(x1).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label, x0, y0, x1, y1)

            if y0 - label_size[1] >= 0:
                text_origin = np.array([x0, y0 - label_size[1]])
            else:
                text_origin = np.array([x0, y0 + 1])

            for i in range(thickness):
                # rectangle param:xy:[x0,y0,x1,y1]
                draw.rectangle([x0 + i, y0 + i, x1 - i, y1 - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])

        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image = img2rgb(image)

        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        image_data = resize_image(image, (self.conf['input_shape'][1], self.conf['input_shape'][0]),
                                  self.conf['letterbox_image'])

        #   添加上batch_size维度

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.cuda() if self.CUDA else images

            #   将图像输入网络当中进行预测！

            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)

            #   将预测框进行堆叠，然后进行非极大抑制

            results = self.bbox_util.nms_(torch.cat(outputs, 1),
                                          self.num_classes,
                                          self.conf['input_shape'],
                                          image_shape,
                                          self.conf['letterbox_image'],
                                          conf_thres=self.conf['confidence'],
                                          nms_thres=self.conf['nms_iou'])

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #   将图像输入网络当中进行预测！

                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)

                #   将预测框进行堆叠，然后进行非极大抑制

                results = self.bbox_util.nms_(torch.cat(outputs, 1),
                                              self.num_classes,
                                              self.conf['input_shape'],
                                              image_shape,
                                              self.conf['letterbox_image'],
                                              conf_thres=self.conf['confidence'],
                                              nms_thres=self.conf['nms_iou'])

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        image_shape = np.array(np.shape(image)[0:2])

        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image = img2rgb(image)

        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        image_data = resize_image(image, (self.conf['input_shape'][1], self.conf['input_shape'][0]),
                                  self.conf['letterbox_image'])

        #   添加上batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.cuda() if self.CUDA else images

            #   将图像输入网络当中进行预测
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)

            #   将预测框进行堆叠，然后进行非极大抑制
            results = self.bbox_util.nms_(torch.cat(outputs, 1),
                                          self.num_classes,
                                          self.conf['input_shape'],
                                          image_shape,
                                          self.conf['letterbox_image'],
                                          conf_thres=self.conf['confidence'],
                                          nms_thres=self.conf['nms_iou'])

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return
