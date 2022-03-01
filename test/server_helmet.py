import os
import time
import json
import base64

import requests
import cv2

url_onnx = "http://192.168.2.165:30002/yolov3_get_helmet_onnx"
url = "http://192.168.2.165:30002/yolov3_get_helmet"


def torch_model(image, image_base64):
    payload = json.dumps({
        "img": [image_base64]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    t1 = time.time()
    response = requests.request("POST", url, headers=headers, data=payload)
    t2 = time.time()
    print(f'torch model time:{round(t2 - t1, 2)}')

    data = response.text
    data = eval(data)
    for item in data['data']:
        x1, y1 = item['left'], item['top']
        x2, y2 = item['left'] + item['width'], item['top'] + item['height']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)


def onnx_model(image, image_base64):
    payload = json.dumps({
        "img": [image_base64]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    t1 = time.time()
    response = requests.request("POST", url_onnx, headers=headers, data=payload)
    t2 = time.time()
    print(f'onnx model use time:{round(t2 - t1, 2)}')
    data = response.text
    data = eval(data)
    for item in data['data']:
        x1, y1 = item['left'], item['top']
        x2, y2 = item['left'] + item['width'], item['top'] + item['height']
        cv2.rectangle(image, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (255, 0, 0), 1)


def video():
    video_path = '/home/ubuntu/github/opencv/samples/data/vtest.avi'
    capture = cv2.VideoCapture(video_path)
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        image = cv2.imencode('.jpg', frame)[1]
        image_base64 = str(base64.b64encode(image))[2:-1]
        # torch_model(frame, image_base64)
        onnx_model(frame, image_base64)

        cv2.imshow('image', frame)
        cv2.waitKey(30)


def image(folder: str):
    imgs = os.listdir(folder)
    for img in imgs:
        img_path = os.path.join(folder, img)
        print(img_path)
        img = cv2.imread(img_path)

        # opencv mat->base64 code
        image = cv2.imencode('.jpg', img)[1]
        image_base64 = str(base64.b64encode(image))[2:-1]

        torch_model(img, image_base64)
        onnx_model(img, image_base64)

        cv2.imshow('image', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    # video()
    image('/home/ubuntu/桌面/images/worker')
