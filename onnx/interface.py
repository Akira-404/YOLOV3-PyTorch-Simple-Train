import os
import onnx
import onnxruntime
import numpy as np
import torch
from utils.utils_prediect import Predict
from PIL import Image

onnx_model = './person.onnx'
model = onnx.load(onnx_model)  # 加载onnx
onnx.checker.check_model(model)  # 检查生成模型是否错误

assert os.path.exists(onnx_model)
f'{onnx_model} is not found.'

session = onnxruntime.InferenceSession(onnx_model)
predict = Predict('../predict.yaml', load_weights=False)

image = Image.open('../person.jpeg')
image_data, image_shape = predict.preprocess(image)
print(np.shape(image_data))
outputs = session.run(None, {'input': image_data})

decode_data = []
for item in outputs:
    decode_data.append(torch.tensor(item))

decode_data = tuple(decode_data)
ret, image = predict.decode(decode_data, image, image_shape)

print(ret)
# print(type(image))
image.show()
