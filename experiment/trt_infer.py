import os
import time

import torch
import torchvision
from PIL import Image
import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from utils.utils_bbox import DecodeBox
from utils.utils_image import image_preprocess
from utils.utils import load_yaml_conf, get_classes, get_anchors

# def post_process(output, origin_h=None, origin_w=None):
#     # 获取检测到框的个数
#     num = int(output[0])
#     # Reshape to a two dimentional ndarray
#     pred = np.reshape(output[1:], (-1, 6))[:num, :]
#     # 转换为torch张量
#     pred = torch.Tensor(pred).cuda()
#     # 框
#     boxes = pred[:, :4]
#     # 置信度
#     scores = pred[:, 4]
#     # classid
#     classid = pred[:, 5]
#     # 根据 score > CONF_THRESH 滤除框
#     si = scores > 0.5
#     boxes = boxes[si, :]
#     scores = scores[si]
#     print(boxes)
#     print(scores)
#     print(classid)
# classid = classid[si]
# # [center_x, center_y, w, h] -> [x1, y1, x2, y2]
# boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
# # nms
# indices = torchvision.ops.nms(boxes, scores, iou_threshold=0.5).cpu()
# # 只保留人体信息
# result_classid = classid[indices].cpu()
# idx = indices[result_classid == 0]
# result_boxes = boxes[idx, :].cpu()
# result_scores = scores[idx].cpu()

# return result_boxes, result_scores, result_classid


local_path = os.path.dirname(os.path.dirname(__file__))

conf = load_yaml_conf('../predict.yaml')
type_ = conf['object'][conf['obj_type']]
classes_path = os.path.join(local_path, type_['classes_path'])
class_names, num_classes = get_classes(classes_path)
anchors_path = os.path.join(local_path, type_['anchors_path'])
anchors, num_anchors = get_anchors(anchors_path)

# read the image
image = Image.open('../images/work.jpeg')
w, h = image.size
image_shape = np.array((h, w))
image_data = image_preprocess(image, (conf['input_shape'][0], conf['input_shape'][1]))

cuda.init()

cfx = cuda.Device(0).make_context()
cfx.pop()

stream = cuda.Stream()
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(TRT_LOGGER)

engine_file_path = "../person.engine"
with open(engine_file_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

host_inputs = []
cuda_inputs = []
host_outputs = []
cuda_outputs = []
bindings = []

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    # 分配主机和设备buffers
    host_mem = cuda.pagelocked_empty(size, dtype)  # 主机
    cuda_mem = cuda.mem_alloc(host_mem.nbytes)  # 设备
    # 将设备buffer绑定到设备.
    bindings.append(int(cuda_mem))
    # 绑定到输入输出
    if engine.binding_is_input(binding):
        host_inputs.append(host_mem)  # CPU
        cuda_inputs.append(cuda_mem)  # GPU
    else:
        host_outputs.append(host_mem)
        cuda_outputs.append(cuda_mem)

t1 = time.time()
# 拷贝输入图像到主机buffer
np.copyto(host_inputs[0], image_data.ravel())
# 将输入数据转到GPU.
cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
# 推理.
context.execute_async(bindings=bindings, stream_handle=stream.handle)
# 将推理结果传到CPU.
cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
# 同步 stream
stream.synchronize()
t2 = time.time()
print(f'time:{t2 - t1}')
# 拿到推理结果 batch_size = 1
output = host_outputs[0]
outputs = list([torch.tensor(item) for item in output])

# post_process(output)
# print(np.shape(host_outputs))

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
