import base64
import time
from PIL import Image
from io import BytesIO
import cv2

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/server', methods=['POST'])
def server1():
    params = request.json if request.method == "POST" else request.args
    t1 = time.time()
    img = params['img']
    image = base64_to_pil(img)
    t2 = time.time()
    print(f'get the image time:{t2 - t1}s')

    result = {
        "code": 200,
        "message": 'success',
    }

    return jsonify(result)


# input:image base64 code without head code
# output:PIL image
def base64_to_pil(base64_data):
    img = None
    for i, data in enumerate(base64_data):
        decode_data = base64.b64decode(data)
        img_data = BytesIO(decode_data)
        img = Image.open(img_data)
    return img


def run():
    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', port=5000, use_reloader=False)


if __name__ == "__main__":
    run()
