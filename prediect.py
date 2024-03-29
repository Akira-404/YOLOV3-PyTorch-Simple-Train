import os
import time

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from modules.yolo import YOLO
from modules.yolo_spp import YOLOSPP
from utils.common import get_anchors, get_classes, load_yaml, load_weights
from utils.utils_bbox import DecodeBox
from utils.image import img2rgb, image_normalization, resize_image, image_preprocess



class Predict:
    def __init__(self, conf_path: str, ignore_track: bool = False):
        """
        :param conf_path: xxx.yaml
        """
        super(Predict, self).__init__()
        self.ignore_track = ignore_track

        # 加载配置文件
        self.conf = load_yaml(conf_path)

        self.CUDA = True if torch.cuda.is_available() and self.conf['cuda'] else False
        self.device = torch.device('cuda' if self.CUDA else 'cpu')

        self.ignore_track = ignore_track
        cwd = os.path.dirname(__file__)

        self.classes_path = os.path.join(cwd, self.conf['classes_path'])
        self.anchors_path = os.path.join(cwd, self.conf['anchors_path'])
        self.model_path = os.path.join(cwd, self.conf['infer_model_path'])

        assert os.path.exists(self.classes_path) is True, self.classes_path
        assert os.path.exists(self.anchors_path) is True, self.anchors_path + 'is error'
        assert os.path.exists(self.model_path) is True, self.model_path

        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)

        self.bbox_util = DecodeBox(self.anchors,
                                   self.num_classes,
                                   (self.conf['image_shape'][0], self.conf['image_shape'][1]),
                                   self.conf['anchors_mask'])

        # load the yolov3 net
        if self.conf['spp']:
            self.net = YOLOSPP(self.conf['anchors_mask'], self.num_classes, self.conf['spp'], self.conf['activation'])
        else:
            self.net = YOLO(self.conf['anchors_mask'], self.num_classes, self.conf['activation'])

        self.prepare_flag = False

    def get_model_with_weights(self, ignore_track: bool = False):
        checkpoint = torch.load(self.model_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        # load_weights(self.net, self.model_path, self.device, ignore_track)
        return self.net

    def get_model_without_weights(self):
        return self.net

    def load_weights(self, ignore_track: bool = False):
        self.net = self.get_model_with_weights(ignore_track)
        self.net = self.net.eval()
        logger.info(f'{self.conf["model_path"]} model,anchors,classes loaded')

        if self.CUDA:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        self.prepare_flag = True

    def detect_image(self, image) -> list:

        if not self.prepare_flag:
            logger.info('This net is not load weights,please use Predict.load_weights()')
            return []

        # image_shape = np.array(np.shape(image)[0:2])  # h,w
        w, h = image.size
        image_shape = np.array((h, w))  # h,w
        image_data = image_preprocess(image, (self.conf['image_shape'][0],
                                              self.conf['image_shape'][1]))

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
                                          self.conf['image_shape'],
                                          image_shape,
                                          self.conf['letterbox_image'],
                                          conf_thres=self.conf['confidence'],
                                          nms_thres=self.conf['nms_iou'])

            if results[0] is None:
                return []

            # top_label=[class_idx1,class_idx2,...]
            top_label = np.array(results[0][:, 6], dtype='int32')

            # top_conf=[conf1,conf2,...]
            top_conf = results[0][:, 4] * results[0][:, 5]

            # top_boxes=[[x1,y1,x2,y2],...]
            top_boxes = results[0][:, :4]

        # if draw:
        #     draw_box(self.num_classes, image, top_label, top_conf, top_boxes, self.class_names,
        #              self.conf['image_shape'])

        # package data
        data = []
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

    def get_fpg(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])

        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image = img2rgb(image)

        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        image_data = resize_image(image, (self.conf['image_shape'][1], self.conf['image_shape'][0]),
                                  self.conf['letterbox_image'])

        #   添加上batch_size维度

        image_data = np.expand_dims(np.transpose(image_normalization(np.array(image_data, dtype='float32')), (2, 0, 1)),
                                    0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.cuda() if self.CUDA else images

            #   将图像输入网络当中进行预测！

            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)

            #   将预测框进行堆叠，然后进行非极大抑制

            results = self.bbox_util.nms_(torch.cat(outputs, 1),
                                          self.num_classes,
                                          self.conf['image_shape'],
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
                                              self.conf['image_shape'],
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
        image_data = resize_image(image, (self.conf['image_shape'][1], self.conf['image_shape'][0]),
                                  self.conf['letterbox_image'])

        #   添加上batch_size维度
        image_data = np.expand_dims(np.transpose(image_normalization(np.array(image_data, dtype='float32')), (2, 0, 1)),
                                    0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.cuda() if self.CUDA else images

            #   将图像输入网络当中进行预测
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)

            #   将预测框进行堆叠，然后进行非极大抑制
            results = self.bbox_util.nms_(torch.cat(outputs, 1),
                                          self.num_classes,
                                          self.conf['image_shape'],
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


