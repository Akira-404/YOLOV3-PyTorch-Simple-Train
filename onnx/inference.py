import torch
import onnxruntime
from utils.utils_prediect import Predict
from PIL import Image

onnx_path = './person.onnx'

session = onnxruntime.InferenceSession(onnx_path)
predict = Predict('../predict.yaml', load_weights=False)

image = Image.open('../person.jpeg')
image_data, image_shape = predict.preprocess(image)
outputs = session.run(None, {'input': image_data})

outputs = tuple([torch.tensor(item) for item in outputs])
ret = predict.decode(outputs, image, image_shape)

for item in ret:
    print(item)
