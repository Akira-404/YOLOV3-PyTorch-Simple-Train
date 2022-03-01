import torch
import onnx
from utils.utils_prediect import Predict

predict = Predict('../predict.yaml')
torch_model = predict.get_model_with_weights()

# export params:
onnx_path = './head.onnx'
batch_size = 1
input_shape = (3, 416, 416)

input_names = ['input']
output_names = ['output']

onnx_simplify = True

dynamic = False
dynamic_params = {
    'input': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
    'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
} if dynamic else None

torch_model = torch_model.eval()
torch_model = torch_model.cuda() if torch.cuda.is_available() else torch_model
input_args = torch.randn(batch_size, *input_shape)
input_args = input_args.cuda() if torch.cuda.is_available() else input_args

torch.onnx.export(model=torch_model,
                  args=input_args,
                  f=onnx_path,
                  opset_version=11,
                  do_constant_folding=True,  # 常量折叠优化
                  export_params=True,
                  input_names=input_names,  # 输出名
                  output_names=output_names,  # 输入名
                  dynamic_axes=dynamic_params)
# Checks
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)  # check onnx model

# Simplify
if onnx_simplify:
    try:
        import onnxsim

        onnx_model, check = onnxsim.simplify(onnx_model,
                                             dynamic_input_shape=dynamic,
                                             input_shapes={'images': list(input_shape)} if dynamic else None)
        assert check, 'assert check failed'
        onnx.save(onnx_model, onnx_path)

    except Exception as e:
        print(e)

print('torch2onnx finish.')
# print(f'input torch model:{torch_model}')
# print(f'output onnx model:{onnx_model}')
print(f'model input shape:{input_shape}')
