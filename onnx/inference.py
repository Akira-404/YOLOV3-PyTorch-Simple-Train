import os
import onnx
import onnxruntime
import numpy as np
import torch
from utils.utils_prediect import Predict
from PIL import Image

onnx_path = './person.onnx'
onnx_model = onnx.load(onnx_path)  # 加载onnx

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
ret = predict.decode(decode_data, image, image_shape)
print(ret)
