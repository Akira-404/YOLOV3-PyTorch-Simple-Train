from yacs.config import CfgNode as CN

_C = CN()

_C.default = CN()
_C.default.root = '/home/ubuntu/pycharmprojects/yolov3-pytorch-simple-train/'
_C.default.year = '2007'
_C.default.cuda = True
_C.default.num_worker = 8  # train
_C.default.batch_size = 1  # prediction
_C.default.input_shape = [416, 416]
_C.default.type = 'person'
_C.default.ssp = []  # spp: [ 6,9,13 ]
_C.default.mosaic = False
_C.default.cosine_lr = True
_C.default.label_smoothing = 0
_C.default.activation = 'leakyrelu'  # support leakyrelu or mish
_C.default.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

# split dataset
_C.default.train_percent = 0.9
_C.default.trainval_percent = 0.9
_C.default.trainval_annotation_path = _C.default.root + '2007_train.txt'
_C.default.val_annotation_path = _C.default.root + '2007_val.txt'

_C.default.freeze_train = False
_C.default.freeze_epoch = 10
_C.default.freeze_batch_size = 16
_C.default.freeze_lr = 1e-4
_C.default.unfreeze_epoch = 120
_C.default.unfreeze_batch_size = 8
_C.default.unfreeze_lr = 1e-4

_C.default.label_smoothing = 0
_C.default.confidence = 0.5  # 置信度
_C.default.nms_iou = 0.3  # nms iou阈值
_C.default.letterbox_image = False  # 是否使用letterbox缩放

_C.object = CN()
_C.object.person = CN()
_C.object.person.dataset_root = '/home/cv/ai_data/person_yolo'
_C.object.person.model_path = 'weights/person.pth'
_C.object.person.classes_path = 'data/person_classes.yaml'
_C.object.person.anchors_path = 'data/person_anchors.yaml'

_C.object.head = CN()
_C.object.head.dataset_root = '/home/cv/ai_data/head_datas_yolo'
_C.object.head.model_path = _C.default.root + 'weights/head.pth'
_C.object.head.classes_path = _C.default.root + 'data/head_classes.yaml'
_C.object.head.anchors_path = _C.default.root + 'data/head_anchors.yaml'

_C.object.helmet = CN()
_C.object.helmet.dataset_root = '/home/cv/ai_data/hardhatworker_voc'
_C.object.helmet.model_path = _C.default.root + 'weights/helmet.pth'
_C.object.helmet.classes_path = _C.default.root + 'data/helmet_classes.yaml'
_C.object.helmet.anchors_path = _C.default.root + 'data/helmet_anchors.yaml'

_C.threshold = CN()
_C.threshold.person = 0.65
_C.threshold.head = 0.65
_C.threshold.helmet = 0.65

_C.http = CN()
_C.http.local = '0.0.0.0'
_C.http.person_port = 30000
_C.http.head_port = 30001
_C.http.helmet_port = 30002

_C.url = CN()
# _C.url.default = 'http://192.168.2.165'  # ai服务器ip地址
_C.url.default = 'http://192.168.2.7'  # ai服务器ip地址
_C.url.smoke = _C.url.default + ":24410/yolov3_get_smoke_onnx"
_C.url.safety_rope = _C.url.default + ":24411/yolov3_get_safety_rope_onnx"
_C.url.cloth = _C.url.default + ":24430/yolov3_get_cloth_onnx"
_C.url.helmet = _C.url.default + ":" + str(_C.http.helmet_port) + "/yolov3_get_helmet_onnx"
_C.url.head = _C.url.default + ":" + str(_C.http.head_port) + "/yolov3_get_head_onnx"
_C.url.person = _C.url.default + ":" + str(_C.http.person_port) + "/yolov3_get_person_onnx"
_C.url.area = _C.url.default + ":" + str(_C.http.person_port) + "/yolov3_poly"

# loguru config
_C.log = CN()
_C.log.file_person = './logs/person/person_{time}.log'
_C.log.file_helmet = './logs/helmet/helmet_{time}.log'
_C.log.file_head = './logs/head/head_{time}.log'
_C.log.level = 'WARNING'
_C.log.rotation = '10 MB'
_C.log.retention = '7 day'
_C.log.compression = 'zip'

_C.freeze()


def get_http():
    return _C.http


def get_object():
    return _C.object


def get_log_config():
    return _C.log


def get_url():
    return _C.url


def get_threshold():
    return _C.threshold


def get_config_defaults():
    return _C.clone()


if __name__ == '__main__':
    config = get_config_defaults()
    print(config)

    t = get_threshold()
    print(t)
