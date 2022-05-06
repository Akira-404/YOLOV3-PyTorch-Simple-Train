import os
import copy

from loguru import logger
from torch.utils.data.dataset import Dataset

from utils.common import load_yaml, read_txt
from utils.image import image_normalization
from utils.data_augmentation import *


class YOLODataset(Dataset):

    # 初始化类
    def __init__(self, config: str, train: bool = True):
        cwd = os.path.dirname(__file__)
        self.config = load_yaml(os.path.join(cwd, config))
        self.istrain = train

        self._root = self.config['train_dataset_root'] if train else self.config['test_dataset_root']

        self.images_path = os.path.join(self._root, 'images')
        self.labels_path = os.path.join(self._root, 'labels')

        self.classes_file_path = os.path.join(self._root, 'classes.names')
        self.train_file_path = os.path.join(self._root, 'train.txt')
        self.test_file_path = os.path.join(self._root, 'test.txt')

        self._use_text = self.config['use_text']
        self._use_mosaic = self.config['use_mosaic']

        logger.info(f'[YOLODataset]::dataset _root: {self._root}')
        logger.info(f'[YOLODataset]::images path: {self.images_path}')
        logger.info(f'[YOLODataset]::labels path: {self.labels_path}')
        logger.info(f'[YOLODataset]::classes file: {self.classes_file_path}')
        logger.info(f'[YOLODataset]::train.txt path: {self.train_file_path}')
        logger.info(f'[YOLODataset]::test.txt path: {self.test_file_path}')

        # 读取train.txt中内容
        self.img_ids = read_txt(self.train_file_path)

        # ['000009', '000052']
        self.img_ids = [x.strip() for x in self.img_ids]
        logger.info(f'[YOLODataset]::image len: {len(self.img_ids)}')

        # read the classes.names
        with open(self.classes_file_path, 'rt') as f:
            self.classes_names = f.read().rstrip("\n").split("\n")

        self.name2id = dict(zip(self.classes_names, range(len(self.classes_names))))
        logger.info(f'[YOLODataset]::[name:id]: {self.name2id}')

        logger.info("Dataset init finished  ! !")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        index = index % len(self.img_ids)
        img_id = self.img_ids[index]
        img_file = os.path.join(self.images_path, img_id)
        label_file = img_file.replace('images', 'labels').replace('jpg', 'txt')

        image, boxes = get_random_data(img_path=img_file,
                                       label_path=label_file,
                                       resize=tuple(self.config['image_shape']),
                                       random=self.istrain)

        image = image_normalization(np.array(image, dtype=np.float32))
        image = np.transpose(image, (2, 0, 1))  # (h,w,c)->(c,h,w)

        if len(boxes) != 0:
            # This part is from darknet yolov3
            # darknete box=[xmin,xmax,ymin,ymax]
            # dw = 1. / (size[0])
            # dh = 1. / (size[1])
            # x = (box[0] + box[1]) / 2.0 - 1
            # y = (box[2] + box[3]) / 2.0 - 1
            # w = box[1] - box[0]
            # h = box[3] - box[2]
            # x = x * dw
            # w = w * dw
            # y = y * dh
            # h = h * dh

            boxes_tmp = copy.deepcopy(boxes)
            # this box =[xmin,ymin,xmax,ymax]
            # [cx,cy.]=([x1,y1]+[x2,y2])/2
            boxes[:, [0, 1]] = (boxes_tmp[:, [0, 1]] + boxes_tmp[:, [2, 3]]) / 2 - 1
            # [w,h]=[x2,y2]-[x1,y1]
            boxes[:, [2, 3]] = boxes_tmp[:, [2, 3]] - boxes_tmp[:, [0, 1]]

            # [cx,w]/iw
            # [cy,h]/ih
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.config['image_shape'][1]  # w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.config['image_shape'][0]  # h
            # return box=[cx,cy,w,h]
        return image, boxes


def get_random_data(img_path: str = None,
                    label_path: str = None,
                    resize: tuple = None,
                    hue: float = .1,
                    sat: float = .7,
                    val: float = .4,
                    random: bool = False):
    # read the image
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # read the label

    # format:类别序号 中心横坐标和宽度比 中心纵坐标和高度比 bbox宽度和image宽度之比 bbox高度和image高度之比
    h, w, c = img.shape
    # yolo标注数据文件名为786_rgb_0616.txt
    # bbox_data = read_txt(label_path)
    with open(label_path, 'r') as f:
        bbox_data = f.readlines()

    boxes = []
    for data in bbox_data:
        data = data.split()
        x_, y_, w_, h_ = eval(data[1]), eval(data[2]), eval(data[3]), eval(data[4])

        x1 = w * x_ - 0.5 * w * w_
        x2 = w * x_ + 0.5 * w * w_
        y1 = h * y_ - 0.5 * h * h_
        y2 = h * y_ + 0.5 * h * h_

        box = [x1, y1, x2, y2, data[0]]

        boxes.append(box)

    boxes = np.array(boxes, dtype=np.float32)

    # data augmentation
    # train:random=True
    # val & test :random=False

    # img resize
    # img_resize = image_resize_letterbox(img, resize) if random else img
    img_resize = letterbox(img, resize) if random else img

    # image flip
    flip = rand() < .5 if random else False
    img_resize = image_flip(img_resize) if flip else img_resize

    # change bbox
    boxes = label_adjust(img.shape[:2], resize, boxes, flip) if random else boxes

    # RGB->HSV->RGB
    img_resize = color_jittering(img_resize, hue, sat, val)

    return img_resize, boxes


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes


if __name__ == '__main__':
    dataset = YOLODataset('data/traffic_signs/config.yaml')  # 实例化一个对象
    image, boxes = dataset[0]  # 返回第一张图像及box和对应的类别
    logger.info(image.shape)
    logger.info(boxes)

    img = (image * 255).astype(np.uint8).transpose(1, 2, 0)  # 注意由于图像像素分布0-255，所以转成uint8
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # [cx,w]*iw
    # [cy,h]*ih
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * img.shape[1]
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * img.shape[0]

    # [h,w]=[h/2,w/2]
    boxes[:, 2:4] = boxes[:, 2:4] / 2
    tmp = copy.deepcopy(boxes)

    # [x1,,y1]=[cx-w/2,cy-h/2]
    boxes[:, 0:2] = tmp[:, 0:2] - tmp[:, 2:4]
    # [x2,,y2]=[cx+w/2,cy+h/2]
    boxes[:, 2: 4] = tmp[:, 0:2] + tmp[:, 2:4]

    for box in boxes:
        x1, y1 = int(box[0]), int(box[1])
        x2, y2 = int(box[2]), int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1, 1)
    cv2.imshow('test', img)
    cv2.waitKey(0)
