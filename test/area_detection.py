import json
import time
import base64

import cv2
import requests

import config

_Url = config.get_url()

# bgr
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def draw_area(path: str):
    poly = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(frame, (x, y), 3, BLUE, -1, -1)
            cv2.imshow('area', frame)
            print('add point:', x, y)
            poly.append((x, y))

    cap = cv2.VideoCapture(path)
    flag, frame = cap.read()
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高

    cv2.imshow('area', frame)
    cv2.setMouseCallback("area", click_event)
    cv2.waitKey(0)
    cv2.destroyWindow('area')

    polys = [poly]
    print(polys)

    # fps = cap.get(cv2.CAP_PROP_FPS)
    # # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # # out_cat = cv2.VideoWriter("area_detection.mp4", fourcc, fps, (width, height), True)  # 保存位置/格式
    # while flag:
    #
    #     image = cv2.imencode('.jpg', frame)[1]
    #     image_base64 = str(base64.b64encode(image))[2:-1]
    #     payload = json.dumps({
    #         "img": [image_base64],
    #         "polys": polys
    #     })
    #     headers = {
    #         'Content-Type': 'application/json'
    #     }
    #     t1 = time.time()
    #     response = requests.request("POST", _Url.area, headers=headers, data=payload)
    #     t2 = time.time()
    #     print(f'arae time:{round(t2 - t1, 2)}')
    #
    #     res_data = eval(response.text)
    #     if res_data['code'] != 200:
    #         print(f'res data::code={res_data["code"]}')
    #         return
    #
    #     data = res_data['data']
    #
    #     for item in data:
    #         # if item['score'] > t:
    #         foot_x = int(item['left'] + item['width'] * 0.5)
    #         foot_y = int(item['top'] + item['height'])
    #
    #         cv2.circle(frame, (foot_x, foot_y), 2, (0, 0, 255), 3, -1)
    #         cv2.putText(frame, 'inside', (item['left'], item['top']), cv2.FONT_HERSHEY_PLAIN, 1, RED)
    #         cv2.rectangle(frame,
    #                       (item['left'], item['top']),
    #                       ((item['left'] + item['width']), (item['top'] + item['height'])), RED, 2)
    #
    #     for poly in polys:
    #         poly_len = len(poly)
    #         limit = lambda x: x % poly_len
    #         for i in range(len(poly)):
    #             start_p = tuple(poly[limit(i)])
    #             end_p = tuple(poly[limit(i + 1)])
    #             cv2.line(frame, start_p, end_p, BLUE, 2)
    #
    #     # out_cat.write(frame)  # 保存视频
    #
    #     cv2.imshow('video', frame)
    #     cv2.waitKey(25)
    #     flag, frame = cap.read()
    #
    # # out_cat.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    draw_area('/home/ubuntu/github/opencv/samples/data/vtest.avi')
