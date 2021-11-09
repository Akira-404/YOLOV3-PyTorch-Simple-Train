import numpy as np
from torch.utils.data.dataset import Dataset
import cv2
from PIL import Image
from utils.utils import img2rgb, preprocess_input


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes: int, train):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.length = len(self.annotation_lines)

    def __len(self) -> int:
        return self.length

    def __getitem__(self, item):
        ...


def rand(a=0, b=1) -> np.ndarray:
    return np.random.rand() * (b - a) + a


def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes
