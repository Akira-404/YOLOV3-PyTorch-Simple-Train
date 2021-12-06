import base64
from PIL import Image
from utils.utils_prediect import Predict
from flask import Flask, jsonify, request
from io import BytesIO
from utils.polygon import winding_number

predict = Predict('../predict.yaml', 'person')

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
