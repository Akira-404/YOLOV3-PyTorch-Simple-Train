import base64
from PIL import Image

import onnxruntime
import torch

from utils.utils import load_yaml_conf, get_classes, get_anchors
from utils.utils_bbox import DecodeBox
from utils.utils_image import image_preprocess
from utils.utils_prediect import Predict
from flask import Flask, jsonify, request
from io import BytesIO
from utils.polygon import winding_number
import numpy as np

predict = Predict('predict.yaml', obj_type='person')

app = Flask(__name__)


def _pil_to_base64(img):
    # img = Image.open(image_path)
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str


def _base64_to_pil(base64_data):
    img = None
    for i, data in enumerate(base64_data):
        decode_data = base64.b64decode(data)
        img_data = BytesIO(decode_data)
        img = Image.open(img_data)
        # data.append(img)
    return img


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
    params = request.json if request.method == "POST" else request.args
    img = _base64_to_pil(params['img'])
    data = predict.tiny_detect_image(img)
    t = 0.65
    post_data = [item for item in data if item['score'] > t]
    return _get_result(200, 'success', post_data)


# TODO:need to debug
@app.route('/yolov3_get_person2', methods=['POST'])
def get_person2():
    params = request.json if request.method == "POST" else request.args
    img = _base64_to_pil(params['img'])
    image, data = predict.detect_image(img, draw=False)
    t = 0.65
    post_data = [item for item in data if item['score'] > t]
    return _get_result(200, 'success', post_data)


conf = load_yaml_conf('./predict.yaml')
type_ = conf['object'][conf['obj_type']]
class_names, num_classes = get_classes(type_['classes_path'])
anchors, num_anchors = get_anchors(type_['anchors_path'])
onnx_path = './onnx/person.onnx'
session = onnxruntime.InferenceSession(onnx_path)


@app.route('/yolov3_get_person_onnx', methods=['POST'])
def get_person_onnx():
    params = request.json if request.method == "POST" else request.args
    image = _base64_to_pil(params['img'])
    w, h = image.size
    print(w, h)
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
            exit()

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
    t = 0.65
    post_data = [item for item in data if item['score'] > t]
    return _get_result(200, 'success', post_data)


@app.route('/yolov3_poly', methods=['POST'])
def poly():
    params = request.json if request.method == "POST" else request.args
    try:
        image = _base64_to_pil(params['image'])
        polys = params['polys']
    except Exception as e:
        print(f'yolov3_poly:e:{e}')
        return _get_result(500, 'Error', ["Input data error"])

    t = 0.64
    post_data = []

    if poly:
        data = predict.tiny_detect_image(image)
        for item in data:
            flag = False
            if item['score'] > t:
                foot_x = int(item['left'] + item['width'] * 0.5)
                foot_y = int(item['top'] + item['height'])

                # flag = crossing_number([foot_x, foot_y], polys)
                flag = winding_number((foot_x, foot_y), polys)

            if flag:
                post_data.append(item)

        # api_test('/home/cv/AI_Data/person.avi', predict, polys)

    return _get_result(200, 'success', post_data)


def run():
    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', port=30000, use_reloader=False)


if __name__ == "__main__":
    run()
    # img = Image.open('img.jpg')
    # base54_data = _pil_to_base64(img)
    # print(type(base54_data))
    # _base64_to_pil([base54_data])
