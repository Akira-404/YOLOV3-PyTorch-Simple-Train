from yacs.config import CfgNode as CN

# This file is not yet use

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

_C.lee_config = CN()

_C.object = CN()
_C.object.person = CN()
_C.object.person.dataset_root = '/home/cv/ai_data/person_yolo'
_C.object.person.model_path = _C.default.root + 'weights/person.pth'
_C.object.person.classes_path = _C.default.root + 'data/person_classes.yaml'
_C.object.person.anchors_path = _C.default.root + 'data/person_anchors.yaml'

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

_C.freeze()


def get_config_defaults():
    return _C.clone()


if __name__ == '__main__':
    config = get_config_defaults()
    print(type(config))
