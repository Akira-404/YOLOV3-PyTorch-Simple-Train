import cv2
import numpy as np
from PIL import Image
from utils.utils_prediect import Predict
from utils.utils import load_yaml_conf


def isRayIntersectsSegment(point, start_p, end_p):  # [x,y] [lng,lat]
    # 输入：判断点，边起点，边终点，都是[lng,lat]格式数组
    if start_p[1] == end_p[1]:  # 排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if start_p[1] > point[1] and end_p[1] > point[1]:  # 线段在射线上边
        return False
    if start_p[1] < point[1] and end_p[1] < point[1]:  # 线段在射线下边
        return False
    if start_p[1] == point[1] and end_p[1] > point[1]:  # 交点为下端点，对应spoint
        return False
    if end_p[1] == point[1] and start_p[1] > point[1]:  # 交点为下端点，对应epoint
        return False
    if start_p[0] < point[0] and end_p[0] < point[0]:  # 线段在射线左边
        return False

    xseg = end_p[0] - (end_p[0] - start_p[0]) * (end_p[1] - point[1]) / (end_p[1] - start_p[1])  # 求交
    if xseg < point[0]:  # 交点在射线起点的左侧
        return False
    return True  # 排除上述情况之后


def isPoiWithinPoly(point, poly):
    # 输入：点，多边形三维数组
    # poly=[[[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]],[[w1,t1],……[wk,tk]]] 三维数组

    # 可以先判断点是否在外包矩形内
    # if not isPoiWithinBox(point,mbr=[[0,0],[180,90]]): return False
    # 但算最小外包矩形本身需要循环边，会造成开销，本处略去
    sinsc = 0  # 交点个数
    for epoly in poly:  # 循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]
        for i in range(len(epoly) - 1):  # [0,len-1]
            start_p = epoly[i]
            end_p = epoly[i + 1]
            if isRayIntersectsSegment(point, start_p, end_p):
                sinsc += 1  # 有交点就加1

    return True if sinsc % 2 == 1 else False


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
    import time
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
                flag = isPoiWithinPoly([foot_x, foot_y], [area_point])

                box = RED if flag else GREEN
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
    read_video('/home/cv/AI_Data/person.avi')
