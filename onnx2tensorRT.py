import tensorrt as trt
from loguru import logger
from pathlib import Path
import pkg_resources as pkg
import onnx
import torch
import onnxsim


def export_engine(onnx, file, workspace: int = 4, verbose=False):
    """
    model:onnx model
    im:input data [type:tensor]
    file:engine output file [type:str]
    train:torch model.mode:train or eval
    simplify:using onnx simplify or not
    """

    try:
        logger.info(f'\nTensorRT: starting export with TensorRT {trt.__version__}...')
        assert onnx.exists(), f'failed to export ONNX file: {onnx}'

        f = file.with_suffix('.engine')  # TensorRT engine file
        trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(trt_logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30

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

        return f
    except Exception as e:
        logger.info(f'\nTensorRT: export failure: {e}')


if __name__ == '__main__':
    onnx = Path('')
    engine = Path('')
    export_engine(onnx, engine, verbose=True)
