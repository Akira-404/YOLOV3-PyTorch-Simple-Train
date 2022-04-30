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


def image_resize_letterbox(image_src: np.ndarray = None,
                           dst_size: tuple = None,
                           pad_color: tuple = (114, 114, 114)):
    """
    缩放图片，保持长宽比。
    :param image_src:       原图（numpy）
    :param dst_size:        （h，w）
    :param pad_color:       填充颜色，默认是灰色
    :return:
    """
    src_h, src_w = image_src.shape[:2]
    dst_h, dst_w = dst_size
    scale = min(dst_h / src_h, dst_w / src_w)
    new_h, new_w = int(round(src_h * scale)), int(round(src_w * scale))

    min_edge = min(new_h, new_w)
    pad = dst_h - min_edge

    # yolov5 letterbox version
    # pad = np.mod(max_edge - min_edge, 32)

    if image_src.shape[0:2] != (new_w, new_h):
        image_dst = cv2.resize(image_src, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        image_dst = image_src

    if dst_h == new_w:
        top, down = pad // 2, pad // 2
        left, right = 0, 0
    else:
        top, down = 0, 0
        left, right = pad // 2, pad // 2

    # add edge border
    image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, cv2.BORDER_CONSTANT, value=pad_color)

    return image_dst


if __name__ == "__main__":
    image_path = '/Users/lee/Desktop/dataset/voc/VOCtrain/VOC2007/JPEGImages/000005.jpg'
    im = cv2.imread(image_path)
    boxes = [[263, 211, 324, 339],
             [165, 264, 253, 372],
             [241, 194, 295, 299]]
    boxes = np.array(boxes, dtype=np.float32)
    boxes = np.reshape(boxes, (-1, 4))

    im_resize = image_resize_letterbox(im, (416, 416))
    im_resize = image_flip(im_resize)
    boxes = label_adjust(im.shape[:2], (416, 416), boxes, True)
    print(boxes)

    for box in boxes:
        x1, y1 = int(box[0]), int(box[1])
        x2, y2 = int(box[2]), int(box[3])
        cv2.rectangle(im_resize, (x1, y1), (x2, y2), (0, 0, 255), 1, 1)
    cv2.imshow('img', im_resize)
    cv2.waitKey(0)
