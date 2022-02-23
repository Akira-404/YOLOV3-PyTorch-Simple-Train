class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        scores = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return scores


def to_numpy(tensor):
    # print(tensor.device)

    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def interface_model():
    r_model_path = "./mnist.onnx"
    time_start1 = time.time()
    rnet1 = ONNXModel(r_model_path)
    time_end2 = time.time()
    print('load model cost', time_end2 - time_start1)
    # 测时间
    test_correct = 0
    time_start = time.time()
    for img, label in test_loader:
        img, lable = Variable(img), Variable(label)
        out = rnet1.forward(to_numpy(img))
        pred = np.argmax(out[0][0])
        correct = 0
        if label.item() == pred:
            correct += 1
        test_correct += correct
    time_end = time.time()
    print("[{}/{}]".format(test_correct, len(test_datasets)))
    print('infer cost', time_end - time_start)

