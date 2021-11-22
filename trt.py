import os
import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
TRT_LOGGER = trt.Logger()
# onnx_path = 'model_data/head.onnx'
# engine_file_path = 'model_data/head.trt'

if __name__ == '__main__':
    parse = argparse.ArgumentParser('TensorRT config')
    parse.add_argument('--onnx_path', type=str, default='model_data/head.onnx',
                       help='onnx model path')
    parse.add_argument('--trt_path', type=str, default='model_data/head.trt',
                       help='tensorrt model path')
    args = parse.parse_args()
    assert os.path.exists(args.onnx_path) is True, f'{args.onnx_path} is error'
    print(args)

    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network,
                                                                                                               TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 28  # 256MiB
        builder.max_batch_size = 1

        # Parse model file
        if not os.path.exists(args.onnx_path):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(args.onnx_path))
            exit(0)
        print('Loading ONNX file from path {}...'.format(args.onnx_path))
        with open(args.onnx_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        network.get_input(0).shape = [1, 3, 300, 400]
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(args.onnx_path))
        # network.mark_output(network.get_layer(network.num_layers-1).get_output(0))
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")
        with open(args.trt_path, "wb") as f:
            f.write(engine.serialize())
