import tensorrt as trt
from loguru import logger
from pathlib import Path
import pkg_resources as pkg
import onnx
import torch
import onnxsim


def file_size(path):
    # Return file/dir size (MB)
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'  # string
    if hard:
        assert result, s  # assert min requirements met
    if verbose and not result:
        logger.warning(s)
    return result


def export_onnx(model, im, file, opset, train, dynamic, simplify):
    """
    YOLOv5 ONNX export

    torch model -> onnx model

    model:torch model
    im:input data [type:tensor]
    file:onnx output file [type:Path(str)]
    opset:onnx opset version [type:int]
    train:torch model.mode:train or eval [type:bool]
    dynamic:output dynamic shape [type:bool]
    """
    try:

        logger.info(f'\nONNX: starting export with onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')

        torch.onnx.export(model, im, f, verbose=False, opset_version=opset,
                          training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=not train,
                          input_names=['images'],
                          output_names=['output'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                                        } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # LOGGER.info(onnx.helper.printable_graph(model_onnx.graph))  # print

        # Simplify
        if simplify:
            try:
                logger.info(f'ONNX: simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                logger.info(f'ONNX: simplifier failure: {e}')
        logger.info(f'ONNX: export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        logger.info(f'ONNX: export failure: {e}')


def export_engine(model, im, file, train, simplify, workspace=4, verbose=False):
    """
    YOLOv5 TensorRT export https://developer.nvidia.com/tensorrt

    torch model -> onnx model -> tensorrt engine

    model:torch model
    im:input data [type:tensor]
    file:engine output file [type:Path(str)]
    train:torch model.mode:train or eval
    simplify:using onnx simplify or not
    """

    try:
        print(f'trt.__version__:{trt.__version__}')

        if trt.__version__[0] == '7':  # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
            grid = model.model[-1].anchor_grid
            model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
            export_onnx(model, im, file, 12, train, False, simplify)  # opset 12
            model.model[-1].anchor_grid = grid
        else:  # TensorRT >= 8
            check_version(trt.__version__, '8.0.0', hard=True)  # require tensorrt>=8.0.0
            export_onnx(model, im, file, 13, train, False, simplify)  # opset 13
        onnx = file.with_suffix('.onnx')

        logger.info(f'\nTensorRT: starting export with TensorRT {trt.__version__}...')
        assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
        assert onnx.exists(), f'failed to export ONNX file: {onnx}'
        f = file.with_suffix('.engine')  # TensorRT engine file
        trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(trt_logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, trt_logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f'failed to load ONNX file: {onnx}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        logger.info(f'TensorRT: Network Description:')
        for inp in inputs:
            logger.info(f'TensorRT:\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
        for out in outputs:
            logger.info(f'TensorRT:\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')

        logger.info(f'TensorRT: building FP{16 if builder.platform_has_fast_fp16 else 32} engine in {f}')
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
            t.write(engine.serialize())
        logger.info(f'TensorRT: export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        logger.info(f'\nTensorRT: export failure: {e}')


if __name__ == '__main__':
    from utils.utils_prediect import Predict

    predict = Predict('./predict.yaml')
    torch_model = predict.get_model_with_weights()

    # export params:
    onnx_path = Path('./person.onnx')
    engine_path = Path('./person.engine')
    batch_size = 1
    input_shape = (3, 416, 416)

    input_names = ['input']
    output_names = ['output']

    simplify = True

    dynamic = False
    dynamic_params = {
        'input': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
        'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
    } if dynamic else None

    torch_model = torch_model.eval()
    torch_model = torch_model.cuda() if torch.cuda.is_available() else torch_model
    input_args = torch.randn(batch_size, *input_shape)
    input_args = input_args.cuda() if torch.cuda.is_available() else input_args

    # export_onnx(torch_model, input_args, onnx_path, 12, False, True, True)
    export_engine(torch_model, input_args, engine_path, False, True)
