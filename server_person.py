import json

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

_Thr = config.get_threshold()
_Url = config.get_url()

predict = Predict('predict.yaml', obj_type='person')
# predict.load_weights()
app = Flask(__name__)

conf = load_yaml_conf('./predict.yaml')
type_ = conf['object']['person']
class_names, num_classes = get_classes(type_['classes_path'])
anchors, num_anchors = get_anchors(type_['anchors_path'])

logger.info('Load yolov3 onnx model.')
onnx_path = './onnx/person.onnx'
session = onnxruntime.InferenceSession(onnx_path, providers=onnxruntime.get_available_providers())
logger.info('Load done.')


def _get_result(code: int, message: str, data):
    result = {
        "code": code,
        "message": message,
        "data": data
    }
    print("Response data:", result)
    return jsonify(result)


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
    return _get_result(200, 'success', data)


@app.route('/yolov3_get_person_onnx', methods=['POST'])
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

    outputs = session.run(None, {'input': image_data})
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
            return _get_result(200, 'empty', [])

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
            'score': float(score),
            'height': int(y1 - y0),
            'left': int(x0),
            'top': int(y0),
            'width': int(x1 - x0)
        }
        data.append(item)
    return _get_result(200, 'success', data)


@app.route('/yolov3_poly', methods=['POST'])
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
        return _get_result(500, f'Error:{e}', [])

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
            return _get_result(res_data['code'], 'error:from url_person', [])

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

    return _get_result(200, 'success', post_data)


def run():
    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', port=30000, use_reloader=False)


if __name__ == "__main__":
    run()
