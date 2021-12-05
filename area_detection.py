import time
from typing import Union, List, Tuple
import cv2
import numpy as np
from PIL import Image

from utils.utils_prediect import Predict
from utils.utils import load_yaml_conf
from polygon import crossing_number, winding_number

# bgr

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def api_test(path: str, model, polys: List):
    cap = cv2.VideoCapture(path)
    flag, frame = cap.read()

    fps = cap.get(cv2.CAP_PROP_FPS)
    t = 0.65
    while flag:

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        data = model.tiny_detect_image(image)

        # draw the poly
        for poly in polys:

            poly_len = len(poly)
            limit = lambda x: x % poly_len
            for i in range(poly_len):
                start_p = poly[limit(i)]
                end_p = poly[limit(i + 1)]
                cv2.line(frame, start_p, end_p, BLUE, 2)

        for item in data:
            if item['score'] > t:
                foot_x = int(item['left'] + item['width'] * 0.5)
                foot_y = int(item['top'] + item['height'])

                cv2.circle(frame, (foot_x, foot_y), 2, (0, 0, 255), 3, -1)

                # flag = crossing_number([foot_x, foot_y], polys)
                flag = winding_number((foot_x, foot_y), polys)

                status = 'inside' if flag else 'outside'

                box = RED if flag else GREEN
                cv2.putText(frame, status, (item['left'], item['top']), cv2.FONT_HERSHEY_PLAIN, 1, box)
                cv2.rectangle(frame,
                              (item['left'], item['top']),
                              ((item['left'] + item['width']), (item['top'] + item['height'])), box, 2)

        cv2.imshow('frame', frame)
        cv2.waitKey(int(1000 / fps))
        flag, frame = cap.read()
    cv2.destroyAllWindows()


def draw_area(path: str):
    predict = Predict('predict.yaml', 'person')

    poly = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(frame, (x, y), 3, BLUE, -1, -1)
            cv2.imshow('area', frame)
            print('add point:', x, y)
            poly.append((x, y))

    cap = cv2.VideoCapture(path)
    flag, frame = cap.read()
    cv2.imshow('area', frame)
    cv2.setMouseCallback("area", click_event)
    cv2.waitKey(0)
    cv2.destroyWindow('area')

    print(poly)
    polys = [poly]
    fps = cap.get(cv2.CAP_PROP_FPS)
    t = 0.65
    while flag:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        data = predict.tiny_detect_image(image)

        for item in data:
            if item['score'] > t:
                foot_x = int(item['left'] + item['width'] * 0.5)
                foot_y = int(item['top'] + item['height'])

                cv2.circle(frame, (foot_x, foot_y), 2, (0, 0, 255), 3, -1)

                # outside=0
                # inside=1
                # flag = crossing_number([foot_x, foot_y], polys)
                flag = winding_number((foot_x, foot_y), polys)
                status = 'inside' if flag else 'outside'

                box = RED if flag else GREEN
                cv2.putText(frame, status, (item['left'], item['top']), cv2.FONT_HERSHEY_PLAIN, 1, box)
                cv2.rectangle(frame,
                              (item['left'], item['top']),
                              ((item['left'] + item['width']), (item['top'] + item['height'])), box, 2)
        for i, data in enumerate(poly):
            cv2.line(frame, poly[i % len(poly)], poly[(i + 1) % len(poly)], BLUE, 2)

        cv2.imshow('video', frame)
        cv2.waitKey(int(1000 / fps))
        flag, frame = cap.read()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # read_video('/home/cv/AI_Data/CUHKSquare.mpg')
    draw_area('D:/ai_data/person.avi')
