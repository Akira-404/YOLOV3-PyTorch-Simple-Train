import os

import onnxruntime
import numpy as np
from PIL import Image
import torch

import config
from utils.utils_bbox import DecodeBox
from utils.utils_image import image_preprocess
from utils.utils import load_yaml_conf, get_classes, get_anchors

# load config file
local_path = os.path.dirname(os.path.dirname(__file__))

conf = load_yaml_conf('../predict.yaml')
type_ = conf['object'][conf['obj_type']]
classes_path = os.path.join('../', type_['classes_path'])
class_names, num_classes = get_classes(classes_path)
anchors_path = os.path.join('../', type_['anchors_path'])
anchors, num_anchors = get_anchors(anchors_path)

# load model
onnx_path = './person.onnx'
session = onnxruntime.InferenceSession(onnx_path, providers=onnxruntime.get_available_providers())

# read the image
image = Image.open('../work.jpeg')
w, h = image.size
image_shape = np.array((h, w))
image_data = image_preprocess(image, (conf['input_shape'][0], conf['input_shape'][1]))

# run onnx model
outputs = session.run(None, {'input': image_data})
# print(np.shape(outputs))
outputs = list([torch.tensor(item) for item in outputs])

# decode result data
decodebox = DecodeBox(anchors,
                      num_classes,
                      input_shape=(conf['input_shape'][0], conf['input_shape'][1]),
                      anchors_mask=conf['anchors_mask'])

with torch.no_grad():
    # outputs shape: (3,batch_size,x,y,w,h,conf,classes)
    outputs = decodebox.decode_box(outputs)

    #   将预测框进行堆叠，然后进行非极大抑制
    # results shape:(len(prediction),num_anchors,4)
    results = decodebox.nms_(torch.cat(outputs, 1),
                             num_classes,
                             conf['input_shape'],
                             image_shape,
                             conf['letterbox_image'],
                             conf_thres=conf['confidence'],
                             nms_thres=conf['nms_iou'])

    if results[0] is None:
        exit()

    top_label = np.array(results[0][:, 6], dtype='int32')
    top_conf = results[0][:, 4] * results[0][:, 5]
    top_boxes = results[0][:, :4]

data = []
for i, c in list(enumerate(top_label)):
    predicted_class = class_names[int(c)]
    box = top_boxes[i]
    score = top_conf[i]

    y0, x0, y1, x1 = box

    y0 = max(0, np.floor(y0).astype('int32'))
    x0 = max(0, np.floor(x0).astype('int32'))
    y1 = min(image.size[1], np.floor(y1).astype('int32'))
    x1 = min(image.size[0], np.floor(x1).astype('int32'))

    item = {
        'label': predicted_class,
        'score': float(score),
        'height': int(y1 - y0),
        'left': int(x0),
        'top': int(y0),
        'width': int(x1 - x0)
    }
    data.append(item)

for item in data:
    print(item)
