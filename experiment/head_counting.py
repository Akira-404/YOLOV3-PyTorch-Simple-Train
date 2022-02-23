import os
import json
import base64
import requests
import cv2
from PIL import Image

from utils.utils_prediect import Predict

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def run(path: str):
    predict = Predict(os.path.join('/home/cv/PycharmProjects/YOLOV3-PyTorch', 'predict.yaml'), 'head')

    cap = cv2.VideoCapture(path)
    flag, frame = cap.read()

    fps = cap.get(cv2.CAP_PROP_FPS)
    t = 0.2
    while flag:

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        data = predict.tiny_detect_image(image)
        for item in data:
            if item['score'] > t:
                cv2.rectangle(frame,
                              (item['left'], item['top']),
                              ((item['left'] + item['width']), (item['top'] + item['height'])), RED, 2)

        cv2.imshow('frame', frame)
        cv2.waitKey(int(1000 / fps))
        flag, frame = cap.read()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print(os.getcwd())
    run('/home/cv/AI_Data/MOT_dataset/MOT20-01-raw.webm')
