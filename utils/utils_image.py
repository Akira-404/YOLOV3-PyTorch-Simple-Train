import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont, Image
import colorsys


def preprocess_input(image):
    image /= 255.0
    return image


def resize_image(image, size: tuple, letterbox_image: bool = False):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def img2rgb(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def preprocess(image, size: tuple, letterbox: bool = False):
    image_shape = np.array(np.shape(image)[0:2])  # h,w
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    image = img2rgb(image)
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    image_data = resize_image(image, size, letterbox)

    #   添加上batch_size维度
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    return image_data, image_shape


def draw_box(nc: int, image, top_label, top_conf, top_boxes, class_names, input_shape: list):
    #   画框设置不同的颜色
    hsv_tuples = [(x / nc, 1., 1.) for x in range(nc)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    #   设置字体与边框厚度
    font = ImageFont.truetype(font='data/simhei.ttf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))

    #   图像绘制
    for i, c in list(enumerate(top_label)):
        predicted_class = class_names[int(c)]
        box = top_boxes[i]
        score = top_conf[i]

        # top, left, bottom, right = box
        y0, x0, y1, x1 = box
        # x0, y0, x1, y1 = box

        y0 = max(0, np.floor(y0).astype('int32'))
        x0 = max(0, np.floor(x0).astype('int32'))
        y1 = min(image.size[1], np.floor(y1).astype('int32'))
        x1 = min(image.size[0], np.floor(x1).astype('int32'))

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        # print(label, x0, y0, x1, y1)

        if y0 - label_size[1] >= 0:
            text_origin = np.array([x0, y0 - label_size[1]])
        else:
            text_origin = np.array([x0, y0 + 1])

        for i in range(thickness):
            # rectangle param:xy:[x0,y0,x1,y1]
            draw.rectangle([x0 + i, y0 + i, x1 - i, y1 - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        del draw
