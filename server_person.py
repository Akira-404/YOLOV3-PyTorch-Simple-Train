import json
import os
import time

import requests
import torch
import numpy as np
import onnxruntime
from loguru import logger
from flask import Flask, jsonify, request

from utils.utils import load_yaml_conf, get_classes, get_anchors
from utils.utils_bbox import DecodeBox
from utils.utils_image import image_preprocess
from utils.utils_prediect import Predict
from utils.polygon import winding_number
from utils.utils_image import base64_to_pil
import config

'''
日志大小上限：10 MB
log位置：./logs/person
写入级别:WARNING
保留时间：7天
压缩格式：ZIP
'''
_LOG = config.get_log_config()

if os.path.exists('./logs/person/') is False:
    os.makedirs('./logs/person/ ')

logger.remove(handler_id=None)  # 不在终端输出文本信息
logger.add(sink=_LOG.file_person,
           level=_LOG.level,
           rotation=_LOG.rotation,
           retention=_LOG.retention,
           compression=_LOG.compression)

_Thr = config.get_threshold()
_Url = config.get_url()

_local_path = os.path.dirname(__file__)
predict_file = os.path.join(_local_path, 'predict.yaml')

predict = Predict(predict_file, obj_type='person')
# predict.load_weights()
app = Flask(__name__)

conf = load_yaml_conf(predict_file)
type_ = conf['object']['person']

classes_path = os.path.join(_local_path, type_['classes_path'])
anchors_path = os.path.join(_local_path, type_['anchors_path'])
class_names, num_classes = get_classes(classes_path)
anchors, num_anchors = get_anchors(anchors_path)

logger.info('Load yolov3 onnx model.')
onnx_path = os.path.join(_local_path, './onnx/person.onnx')
t1 = time.time()
session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
t2 = time.time()
print(f'load onnx model use time:{t2 - t1}')
logger.info('Load done.')


def jsonify_(code: int, message: str, data):
    """
    :param code:server status code 200,500...
    :param message: success or error ...
    :param data: list data
    :return: jsonify data
    """
    result = {
        "code": code,
        "message": message,
        "data": data
    }

    logger.info(result)
    # print("Response data:", result)
    return jsonify(result)


@logger.catch()
@app.route('/yolov3_get_person', methods=['POST'])
def get_person():
    """
    Args:
        img:base64 code without code head
    Returns:
        {
            "code": server status code,
            "data": [
              {
                "height": int
                "label": str,
                "left": int,
                "score": float,
                "top": int,
                "width": int
              }
            ],
            "message": "success"
    """
    params = request.json if request.method == "POST" else request.args
    img = base64_to_pil(params['img'])
    data = predict.detect_image(img, draw=False)
    return jsonify_(200, 'success', data)


@app.route('/yolov3_get_person_onnx', methods=['POST'])
@logger.catch()
def get_person_onnx():
    """
        Args:
            img:base64 code without code head
        Returns:
            {
                "code": server status code,
                "data": [
                  {
                    "height": int
                    "label": str,
                    "left": int,
                    "score": float,
                    "top": int,
                    "width": int
                  }
                ],
                "message": "success"
        """
    params = request.json if request.method == "POST" else request.args
    image = base64_to_pil(params['img'])
    w, h = image.size
    image_shape = np.array((h, w))
    image_data = image_preprocess(image, (conf['input_shape'][0], conf['input_shape'][1]))

    t1 = time.time()
    outputs = session.run(None, {'input': image_data})
    t2 = time.time()
    print(t2 - t1)
    outputs = list([torch.tensor(item) for item in outputs])

    # decode result data

    decodebox = DecodeBox(anchors,
                          num_classes,
                          input_shape=(conf['input_shape'][0], conf['input_shape'][1]),
                          anchors_mask=conf['anchors_mask'])

    with torch.no_grad():
        # outputs shape: (3,batch_size,x,y,w,h,conf,classes)
        outputs = decodebox.decode_box(outputs)

        #   将预测框进行堆叠，然后进行非极大抑制
        # results shape:(len(prediction),num_anchors,4)
        results = decodebox.nms_(torch.cat(outputs, 1),
                                 num_classes,
                                 conf['input_shape'],
                                 image_shape,
                                 conf['letterbox_image'],
                                 conf_thres=conf['confidence'],
                                 nms_thres=conf['nms_iou'])

        if results[0] is None:
            return jsonify_(200, 'empty', [])

        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 4] * results[0][:, 5]
        top_boxes = results[0][:, :4]

    data = []
    for i, c in list(enumerate(top_label)):
        predicted_class = class_names[int(c)]
        box = top_boxes[i]
        score = top_conf[i]

        y0, x0, y1, x1 = box

        y0 = max(0, np.floor(y0).astype('int32'))
        x0 = max(0, np.floor(x0).astype('int32'))
        y1 = min(image.size[1], np.floor(y1).astype('int32'))
        x1 = min(image.size[0], np.floor(x1).astype('int32'))

        item = {
            'label': predicted_class,
            'score': round(float(score), 2),
            'height': int(y1 - y0),
            'left': int(x0),
            'top': int(y0),
            'width': int(x1 - x0)
        }
        data.append(item)

    post_data = jsonify_(200, 'success', data)
    return post_data


@app.route('/yolov3_poly', methods=['POST'])
@logger.catch()
def poly():
    """
        Args:
            img:base64 code without code head
            polys:[[(int, int), (int, int), ...],...]
        Returns:
            {
                "code": server statue code,
                "data": [
                  {
                    "height": int
                    "label": str,
                    "left": int,
                    "score": float,
                    "top": int,
                    "width": int
                  }
                ],
                "message": "success"
        """
    params = request.json if request.method == "POST" else request.args
    try:
        # image = base64_to_pil(params['image'])
        polys: list = params['polys']
    except Exception as e:
        return jsonify_(500, f'Error:{e}', [])

    # t = 0.64
    bias = 0.5
    post_data = []

    if poly:
        # data = predict.detect_image(image, draw=False)
        payload = json.dumps({
            "img": params['img']
        })
        headers = {
            'Content-Type': 'application/json'
        }
        url_person = _Url.person
        response = requests.request("POST", url_person, headers=headers, data=payload)
        res_data = eval(response.text)
        if res_data['code'] != 200:
            return jsonify_(res_data['code'], 'error:from url_person', [])

        data = res_data['data']
        for item in data:
            flag = False
            if item['score'] > _Thr.person:
                foot_x = int(item['left'] + item['width'] * bias)
                foot_y = int(item['top'] + item['height'])

                # flag = crossing_number([foot_x, foot_y], polys)
                flag = winding_number((foot_x, foot_y), polys)

            if flag:
                post_data.append(item)

    post_data = jsonify_(200, 'success', post_data)
    return post_data


_HTTP = config.get_http()


def run():
    app.config['JSON_AS_ASCII'] = False
    app.run(host=_HTTP.local, port=_HTTP.person_port, use_reloader=False)


if __name__ == "__main__":
    run()
