import numpy as np
from PIL import Image


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
