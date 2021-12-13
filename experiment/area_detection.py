import os
import base64
import json
import cv2
import requests
import yaml
from PIL import Image

from utils.utils_prediect import Predict
from utils.utils import load_yaml_conf
from utils.polygon import winding_number

# bgr

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def _api_test_v1(path: str, model, polys: list):
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


def image_to_base64(image_np):
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]

    return image_code


def _api_test_v2(path: str):
    cap = cv2.VideoCapture(path)
    flag, frame = cap.read()

    fps = cap.get(cv2.CAP_PROP_FPS)
    t = 0.65
    while flag:
        decode_image = image_to_base64(frame)

        conf = load_yaml_conf('experiment/area_detection.yaml')
        polys = conf['polys']
        # url = "http://192.168.2.165:30000/yolov3_poly"
        url = "http://192.168.2.7:30000/yolov3_poly"
        payload = json.dumps({
            "image": [decode_image],
            "polys": polys
        })
        headers = {'Content-Type': 'application/json'}
        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.text)
        code = eval(response.text)['code']
        print(code)
        local = eval(response.text)['data']
        for poly in polys:
            poly_len = len(poly)
            limit = lambda x: x % poly_len
            for i in range(poly_len):
                start_p = tuple(poly[limit(i)])
                end_p = tuple(poly[limit(i + 1)])
                cv2.line(frame, start_p, end_p, BLUE, 2)
        for item in local:
            x0 = item['left']
            y0 = item['top']
            x1 = item['left'] + item['width']
            y1 = item['top'] + item['height']
            cv2.rectangle(frame,
                          (x0, y0),
                          (x1, y1), (0, 0, 255), 2)

        cv2.imshow('image', frame)
        cv2.waitKey(30)
        flag, frame = cap.read()


def draw_area(path: str):
    predict = Predict('predict.yaml')

    poly = []
    # def click_event(event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         cv2.circle(frame, (x, y), 3, BLUE, -1, -1)
    #         cv2.imshow('area', frame)
    #         print('add point:', x, y)
    #         poly.append((x, y))

    cap = cv2.VideoCapture(path)
    flag, frame = cap.read()
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高

    # cv2.imshow('area', frame)
    # cv2.setMouseCallback("area", click_event)
    # cv2.waitKey(0)
    # cv2.destroyWindow('area')

    polys = [poly]

    print(polys)
    with open('experiment/area_detection.yaml', 'r', encoding='UTF-8') as f:
        conf = yaml.safe_load(f)

    polys = conf['polys']
    fps = cap.get(cv2.CAP_PROP_FPS)
    t = 0.65
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_cat = cv2.VideoWriter("area_detection.mp4", fourcc, fps, (width, height), True)  # 保存位置/格式
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
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

        for poly in polys:
            poly_len = len(poly)
            limit = lambda x: x % poly_len
            for i in range(len(poly)):
                start_p = tuple(poly[limit(i)])
                end_p = tuple(poly[limit(i + 1)])
                cv2.line(frame, start_p, end_p, BLUE, 2)

        # out_cat.write(frame)  # 保存视频

        cv2.imshow('video', frame)
        cv2.waitKey(int(1000 / fps))
        flag, frame = cap.read()

    out_cat.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # read_video('/home/cv/AI_Data/CUHKSquare.mpg')
    draw_area('/home/cv/AI_Data/person.avi')
    # draw_area('/home/cv/AI_Data/MOT_dataset/MOT20-02-raw.webm')
    # _api_test_v2('/home/cv/AI_Data/person.avi')
