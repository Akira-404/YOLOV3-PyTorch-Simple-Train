import cv2
import numpy as np
from PIL import Image
from utils.utils_prediect import Predict
from utils.utils import load_yaml_conf


def read_video(path: str):
    predict = Predict('predict.yaml', 'person')

    color = (255, 0, 0)
    area_point = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(frame, (x, y), 3, color, -1, -1)
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
        data = predict.tiny_detect_image(image)
        # frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        for item in data:
            if item['score'] > t:
                foot_x = int(item['left'] + item['width'] * 0.5)
                foot_y = int(item['top'] + item['height'])
                cv2.rectangle(frame,
                              (item['left'], item['top']),
                              ((item['left'] + item['width']), (item['top'] + item['height'])), (0, 255, 0), 2)
                cv2.circle(frame, (foot_x, foot_y), 2, (0, 0, 255), 3, -1)

        for i, data in enumerate(area_point):
            # print(area_point[i % len(area_point)], area_point[(i + 1) % len(area_point)])
            cv2.line(frame, area_point[i % len(area_point)], area_point[(i + 1) % len(area_point)], color, 2)

        cv2.imshow('video', frame)
        cv2.waitKey(int(1000 / fps))
        flag, frame = cap.read()
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


def curve1():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    x = np.array([30, 50, 100, 120])
    y = np.array([100, 150, 240, 200])
    for i in range(len(x)):
        cv2.circle(image, (x[i], y[i]), 3, (255, 0, 0), -1, 8, 0)

    poly = np.poly1d(np.polyfit(x, y, 3))
    print(poly)
    for t in range(30, 250, 1):
        y_ = np.int(poly(t))
        cv2.circle(image, (t, y_), 1, (0, 0, 255), 1, 8, 0)
    cv2.imshow("fit curve", image)
    cv2.waitKey(0)


def curve2():
    img1 = cv2.imread('img2.jpg')

    pts = np.array([[300, 100], [320, 300], [500, 310], [600, 450], [650, 600], [700, 680]])  # 随便取几个散点
    pts_fit2 = np.polyfit(pts[:, 0], pts[:, 1], 2)  # 拟合为二次曲线
    pts_fit3 = np.polyfit(pts[:, 0], pts[:, 1], 3)  # 拟合为三次曲线
    print(pts_fit2)  # 打印系数列表，含三个系数
    print(pts_fit3)  # 打印系数列表，含四个系数

    plotx = np.linspace(300, 699, 400)  # 按步长为1，设置点的x坐标
    ploty2 = pts_fit2[0] * plotx ** 2 + pts_fit2[1] * plotx + pts_fit2[2]  # 得到二次曲线对应的y坐标
    ploty3 = pts_fit3[0] * plotx ** 3 + pts_fit3[1] * plotx ** 2 + pts_fit3[2] * plotx + pts_fit3[3]  # 得到三次曲线对应的y坐标

    pts_fited2 = np.array([np.transpose(np.vstack([plotx, ploty2]))])  # 得到二次曲线对应的点集
    pts_fited3 = np.array([np.transpose(np.vstack([plotx, ploty3]))])  # 得到三次曲线对应的点集

    cv2.polylines(img1, [pts], False, (0, 0, 0), 5)  # 原始少量散点构成的折线图
    cv2.polylines(img1, np.int_([pts_fited2]), False, (0, 255, 0), 5)  # 绿色 二次曲线上的散点构成的折线图，近似为曲线
    cv2.polylines(img1, np.int_([pts_fited3]), False, (0, 0, 255), 5)  # 红色 三次曲线上的散点构成的折线图，近似为曲线
    cv2.namedWindow('img1', 0)
    cv2.imshow('img1', img1)

    cv2.waitKey(0)


# def detect_person(path: str):
#     predict = Predict('predict.yaml', 'reflective')
#     conf = load_yaml_conf('predict.yaml')
#     image = Image.open(args.image)
#     ret_image = predict.detect_image(image)


if __name__ == '__main__':
    # draw_area('img2.jpg', (255, 0, 0))
    # read_video('/home/cv/AI_Data/CUHKSquare.mpg')
    read_video('/home/cv/AI_Data/person.avi')
