from loguru import logger
import onnx
import torch
import tensorrt as trt

from utils.utils_prediect import Predict



def export_onnx(model,
                im,
                file: str,
                opset: int = None,
                train: bool = False,
                dynamic: bool = False,
                simplify: bool = True):
    # onnx export arges
    input_names = ['images']
    output_names = ['output']

    dynamic_params = {
        'input': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,h,w)
        'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
    } if dynamic else None

    opset = opset if opset is not None else (12 if trt.__version__ == '7' else 13)
    try:
        torch.onnx.export(model=torch_model,
                          args=input_args,
                          f=file,
                          verbose=False,
                          opset_version=opset,
                          training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=not train,
                          export_params=True,
                          input_names=input_names,  # 输出名
                          output_names=output_names,  # 输入名
                          dynamic_axes=dynamic_params)
        # Checks
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)  # check onnx model

        # Simplify
        if simplify:
            try:
                import onnxsim

                logger.info(f'ONNX: simplifying with onnx-simplifier {onnxsim.__version__}...')
                onnx_model, check = onnxsim.simplify(onnx_model,
                                                     dynamic_input_shape=dynamic,
                                                     input_shapes={'images': list(input_shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(onnx_model, onnx_path)

            except Exception as e:
                logger.error(f'ONNX: simplifier failure: {e}')

            logger.success(f'ONNX: export success, saved as {file})')

    except Exception as e:
        logger.error(f'ONNX: export failure: {e}')


if __name__ == '__main__':
    predict = Predict('../predict.yaml')
    torch_model = predict.get_model_with_weights()

    # model params:
    onnx_path = './head.onnx'
    batch_size = 1
    input_shape = (3, 416, 416)

    torch_model = torch_model.eval()
    torch_model = torch_model.cuda() if torch.cuda.is_available() else torch_model
    input_args = torch.randn(batch_size, *input_shape)
    input_args = input_args.cuda() if torch.cuda.is_available() else input_args

    export_onnx(torch_model, input_args, onnx_path)
