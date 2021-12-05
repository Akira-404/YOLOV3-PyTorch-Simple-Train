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


def read_video(path: str):
    predict = Predict('predict.yaml', 'person')

    area_point = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(frame, (x, y), 3, BLUE, -1, -1)
            cv2.imshow('area', frame)
            print('add point:', x, y)
            area_point.append((x, y))

    cap = cv2.VideoCapture(path)
    flag, frame = cap.read()
    cv2.imshow('area', frame)
    cv2.setMouseCallback("area", click_event)
    cv2.waitKey(0)
    cv2.destroyWindow('area')

    fps = cap.get(cv2.CAP_PROP_FPS)
    t = 0.65
    while flag:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # image = predict.detect_image(image)
        # t1 = time.time()
        data = predict.tiny_detect_image(image)
        # t2 = time.time()
        # print(f'predict time:{t2 - t1}')

        # frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        for item in data:
            if item['score'] > t:
                foot_x = int(item['left'] + item['width'] * 0.5)
                foot_y = int(item['top'] + item['height'])

                cv2.circle(frame, (foot_x, foot_y), 2, (0, 0, 255), 3, -1)

                # outside=0
                # inside=1
                # flag = crossing_number([foot_x, foot_y], [area_point])
                flag = winding_number((foot_x, foot_y), area_point)
                status='inside' if flag else 'outside'

                box = RED if flag else GREEN
                cv2.putText(frame,status,(item['left'],item['top']),cv2.FONT_HERSHEY_PLAIN,1,box)
                cv2.rectangle(frame,
                              (item['left'], item['top']),
                              ((item['left'] + item['width']), (item['top'] + item['height'])), box, 2)
        for i, data in enumerate(area_point):
            # print(area_point[i % len(area_point)], area_point[(i + 1) % len(area_point)])
            cv2.line(frame, area_point[i % len(area_point)], area_point[(i + 1) % len(area_point)], BLUE, 2)

        # t1 = time.time()
        cv2.imshow('video', frame)
        cv2.waitKey(int(1000 / fps))
        flag, frame = cap.read()
        # t2 = time.time()
        # print(f'opencv decode time:{t2 - t1}')
    cv2.destroyAllWindows()


def draw_area(img_path: str, color: tuple = (255, 0, 0)):
    area_point = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 3, color, -1, -1)
            cv2.imshow('original', img)
            print('add point:', x, y)
            area_point.append((x, y))

    img = cv2.imread(img_path)
    cv2.imshow('original', img)
    cv2.setMouseCallback("original", click_event)
    cv2.waitKey(0)
    cv2.destroyWindow('original')
    for i, data in enumerate(area_point):
        print(area_point[i % len(area_point)], area_point[(i + 1) % len(area_point)])
        cv2.line(img, area_point[i % len(area_point)], area_point[(i + 1) % len(area_point)], color, 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # draw_area('img2.jpg', (255, 0, 0))
    # read_video('/home/cv/AI_Data/CUHKSquare.mpg')
    read_video('D:/ai_data/person.avi')
