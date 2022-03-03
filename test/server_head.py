import os
import sys
import time
import json
import base64

import requests
import cv2

sys.path.append("..")
import config

_Url = config.get_url()


def video(path: str = None, draw: bool = True):
    capture = cv2.VideoCapture(path)
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        image = cv2.imencode('.jpg', frame)[1]
        image_base64 = str(base64.b64encode(image))[2:-1]
        payload = json.dumps({
            "img": [image_base64]
        })
        headers = {
            'Content-Type': 'application/json'
        }
        t1 = time.time()
        response = requests.request("POST", _Url.head, headers=headers, data=payload)
        t2 = time.time()
        print(f'model use time:{round(t2 - t1, 2)}')
        data = response.text
        data = eval(data)
        print(data)
        if draw:
            for item in data['data']:
                x1, y1 = item['left'], item['top']
                x2, y2 = item['left'] + item['width'], item['top'] + item['height']
                cv2.rectangle(image, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (255, 0, 0), 1)
                cv2.putText(image, str(item['label']), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
                cv2.putText(image, str(round(item['score'], 3)), (x1 + 60, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
            cv2.imshow('image', frame)
        cv2.waitKey(25)


def image(folder: str, draw: bool = True):
    imgs = os.listdir(folder)
    for img in imgs:
        img_path = os.path.join(folder, img)
        print(img_path)
        img = cv2.imread(img_path)

        # opencv mat->base64 code
        image = cv2.imencode('.jpg', img)[1]
        image_base64 = str(base64.b64encode(image))[2:-1]

        payload = json.dumps({
            "img": [image_base64]
        })
        headers = {
            'Content-Type': 'application/json'
        }
        t1 = time.time()
        response = requests.request("POST", _Url.head, headers=headers, data=payload)
        t2 = time.time()
        print(f'model use time:{round(t2 - t1, 2)}')
        data = response.text
        data = eval(data)
        print(data)
        if draw:
            for item in data['data']:
                x1, y1 = item['left'], item['top']
                x2, y2 = item['left'] + item['width'], item['top'] + item['height']
                cv2.rectangle(image, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (255, 0, 0), 1)
                cv2.putText(image, str(item['label']), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
                cv2.putText(image, str(round(item['score'], 3)), (x1 + 60, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
            cv2.imshow('image', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    # image('/home/ubuntu/桌面/images/person')

    video_path = '/home/ubuntu/github/opencv/samples/data/vtest.avi'
    video(video_path, draw=False)
