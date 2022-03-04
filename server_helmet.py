import os

import torch
import numpy as np
import onnxruntime
from loguru import logger
from flask import Flask, jsonify, request

from utils.utils_prediect import Predict
from utils.utils_image import base64_to_pil
from utils.utils import load_yaml_conf, get_classes, get_anchors
from utils.utils_bbox import DecodeBox
from utils.utils_image import image_preprocess

import config

'''
日志大小上限：10 MB
log位置：./logs/helmet
写入级别:WARNING
保留时间：7天
压缩格式：ZIP
'''
_LOG = config.get_log_config()
_local_path = os.path.dirname(__file__)
log_path = os.path.join(_local_path, 'log/helmet')
if os.path.exists(log_path) is False:
    os.makedirs(log_path)

logger.remove(handler_id=None)  # 不在终端输出文本信息
logger.add(sink=os.path.join(log_path, 'helmet_{time}.log'),
           level=_LOG.level,
           rotation=_LOG.rotation,
           retention=_LOG.retention,
           compression=_LOG.compression)

_local_path = os.path.dirname(__file__)
predict_file = os.path.join(_local_path, 'predict.yaml')
predict = Predict(predict_file, obj_type='helmet')
# predict.load_weights()
app = Flask(__name__)

conf = load_yaml_conf(predict_file)
type_ = conf['object']['helmet']

classes_path = os.path.join(_local_path, type_['classes_path'])
anchors_path = os.path.join(_local_path, type_['anchors_path'])
class_names, num_classes = get_classes(classes_path)
anchors, num_anchors = get_anchors(anchors_path)

logger.info('Load yolov3 onnx model.')
onnx_path = os.path.join(_local_path, './onnx/helmet.onnx')
session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
logger.info('Load done.')


def jsonify_(code: int, message: str, data):
    result = {
        "code": code,
        "message": message,
        "data": data
    }

    return jsonify(result)


@app.route('/yolov3_get_helmet', methods=['POST'])
@logger.catch()
def get_helmet():
    params = request.json if request.method == "POST" else request.args
    img = base64_to_pil(params['img'])
    data = predict.detect_image(img, draw=False)

    return jsonify_(200, 'success', data)


@app.route('/yolov3_get_helmet_onnx', methods=['POST'])
@logger.catch()
def get_helmet_onnx():
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
            'score': float(score),
            'height': int(y1 - y0),
            'left': int(x0),
            'top': int(y0),
            'width': int(x1 - x0)
        }
        data.append(item)
    return jsonify_(200, 'success', data)


_HTTP = config.get_http()


def run():
    app.config['JSON_AS_ASCII'] = False
    app.run(host=_HTTP.local, port=_HTTP.helmet_port, use_reloader=False)


if __name__ == "__main__":
    run()
