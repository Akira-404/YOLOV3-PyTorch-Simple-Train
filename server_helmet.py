import base64
import os
import cv2
import time
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from utils.utils_prediect import Predict
from utils.utils import load_yaml_conf
from flask import Flask, jsonify, request
from io import BytesIO

predict = Predict('predict.yaml', 'helmet')

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


@app.route('/yolov3_get_helmet', methods=['POST'])
def get_person():
    # img = Image.open('img.jpg')
    params = request.json if request.method == "POST" else request.args
    img = _base64_to_pil(params['img'])
    # img.show()
    data = predict.tiny_detect_image(img)
    for i, item in enumerate(data):
        print(i, item)
    return _get_result(200,
                       'success',
                       data)


def run():
    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', port=30002, use_reloader=False)


if __name__ == "__main__":
    run()
    # img = Image.open('img.jpg')
    # base54_data = _pil_to_base64(img)
    # print(type(base54_data))
    # _base64_to_pil([base54_data])
