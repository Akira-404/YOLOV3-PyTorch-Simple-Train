"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/onnx_export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import os
import argparse
import time

import torch
import cv2
import numpy as np
import onnx
import onnxruntime as rt
from PIL import Image
from onnx_simplifier import simplify
from predict import Predict
from utils.utils import load_yaml_conf


def get_onnx_model(model, output_path, input_data):
    model.eval()
    # model[-1].export = True  # set Detect() layer export=True
    y = model(input_data)

    # ONNX export
    try:

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        # f = weights.replace('.pth', '.onnx')  # filename
        torch.onnx.export(model,
                          img,
                          output_path,
                          verbose=False,
                          opset_version=12,
                          input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'],
                          do_constant_folding=True)

        # Checks
        onnx_model = onnx.load(output_path)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        onnx_model, flag = simplify(onnx_model)
        print(f'simplify:{flag}')

        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % output_path)


    except Exception as e:
        print('ONNX export failure: %s' % e)


if __name__ == '__main__':
    # Load conf yaml
    conf = load_yaml_conf('predict.yaml')

    weights = conf['model_path']
    img_size = conf['input_shape']
    batch_size = conf['batch_size']
    onnx_model = conf['onnx_model']
    assert os.path.exists(weights) is True, f'{weights} is error'
    img_size *= 2 if len(img_size) == 1 else 1  # expand

    # Get the model
    predict = Predict('predict.yaml')
    model = predict.get_model()

    # Input
    # image size(1,3,416,416) iDetection
    img = torch.zeros((batch_size, 3, *img_size))
    print(f'Input data shape:{img.shape}')

    get_onnx_model(model, onnx_model, img)
    img = predict.onnx_test('model_data/head.onnx', 'img.jpg', True)
    img.show()
