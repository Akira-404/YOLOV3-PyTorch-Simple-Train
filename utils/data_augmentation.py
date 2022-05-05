import logging

import cv2
import torch
import numpy as np


def rand(a: float = 0.0,
         b: float = 1.0) -> np.ndarray:
    return np.random.rand() * (b - a) + a


# 图片翻转
def image_flip(img: np.ndarray = None,
               flip_code: int = 1) -> np.ndarray:
    return cv2.flip(img, flip_code)


# 色域变换
def color_jittering(image: np.ndarray = None,
                    hue: float = .1,
                    sat: float = .7,
                    val: float = .4) -> np.ndarray:
    image_data = np.array(image, np.uint8)
    r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

    # 将图像转到HSV上
    hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
    dtype = image_data.dtype

    # 应用变换
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
    return image_data


# 坐标调整
def label_adjust(image_shape: tuple = None,
                 resize: tuple = None,
                 boxes: np.ndarray = None,
                 flip: bool = None) -> np.ndarray:
    if len(boxes) > 0:
        ih, iw = image_shape
        dst_h, dst_w = resize
        scale = min(dst_w / iw, dst_h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        pad_w = (dst_w - nw) // 2
        pad_h = (dst_h - nh) // 2

        boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / iw + pad_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / ih + pad_h

        if flip: boxes[:, [0, 2]] = dst_w - boxes[:, [2, 0]]

        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > dst_w] = dst_w
        boxes[:, 3][boxes[:, 3] > dst_h] = dst_h
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]

    return boxes


def letterbox(img,
              new_shape: tuple = (416, 416),
              color: tuple = (114, 114, 114),
              auto: bool = False,
              scaleFill: bool = False,
              scaleup: bool = False):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    img_shape = img.shape[:2]  # current shape [height, width]
    h, w = img_shape

    dst_h, dst_w = new_shape
    # Scale ratio (new / old)
    r = min(dst_h / h, dst_w / w)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(w * r)), int(round(h * r))
    dw, dh = dst_w - new_unpad[0], dst_h - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (dst_w, dst_h)
        ratio = dst_w / w, dst_h / h  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if img_shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img


if __name__ == "__main__":
    image_path = '/Users/lee/Desktop/dataset/voc/VOCtrain/VOC2007/JPEGImages/000005.jpg'
    im = cv2.imread(image_path)
    boxes = [[263, 211, 324, 339],
             [165, 264, 253, 372],
             [241, 194, 295, 299]]
    boxes = np.array(boxes, dtype=np.float32)
    boxes = np.reshape(boxes, (-1, 4))

    im_resize = letterbox(im, (416, 416))
    im_resize = image_flip(im_resize)
    boxes = label_adjust(im.shape[:2], (416, 416), boxes, True)
    print(boxes)

    for box in boxes:
        x1, y1 = int(box[0]), int(box[1])
        x2, y2 = int(box[2]), int(box[3])
        cv2.rectangle(im_resize, (x1, y1), (x2, y2), (0, 0, 255), 1, 1)
    cv2.imshow('img', im_resize)
    cv2.waitKey(0)
