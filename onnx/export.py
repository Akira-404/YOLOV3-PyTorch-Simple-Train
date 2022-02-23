import time
import numpy as np
import os
import torch
import onnx
import onnxruntime

from utils.utils_prediect import Predict

predict = Predict('../predict.yaml')
torch_model = predict.get_model()

onnx_model = './person.onnx'
batch_size = 1
input_shape = (3, 416, 416)

input_names = ['input']
output_names = ['output']

dynamic_params = {
    'input': [2, 3],
    'output': [2, 3]
}
# dynamic_params = None

torch_model = torch_model.eval().cuda() if torch.cuda.is_available() else torch_model
input_args = torch.randn(batch_size, *input_shape)
input_args = input_args.cuda() if torch.cuda.is_available() else input_args

torch.onnx.export(model=torch_model,
                  args=input_args,
                  f=onnx_model,
                  opset_version=11,
                  do_constant_folding=True,  # 常量折叠优化
                  export_params=True,
                  input_names=input_names,  # 输出名
                  output_names=output_names,  # 输入名
                  dynamic_axes=dynamic_params)

print('torch2onnx finish.')
print(f'input torch model:{torch_model}')
print(f'output onnx model:{onnx_model}')
print(f'model input shape:{input_shape}')
