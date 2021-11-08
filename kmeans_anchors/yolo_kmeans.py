import numpy as np


def wh_iou(wh1: np.ndarray, wh2: np.ndarray) -> np.ndarray:
    """
    :param wh1: box1
    :param wh2: box2
    :return: IOU
    """

    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]

    # function:np.prod():根据axis计算所有元素的乘积
    inter = np.minimum(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(axis=2) + wh2.prod(axis=2) - inter)  # iou = inter / (area1 + area2 - inter)


def k_means(boxes: np.ndarray, k: int, dist=np.median) -> np.ndarray:
    """
    refer: https://github.com/qqwweee/keras-yolo3/blob/master/kmeans.py
    :param boxes:需要聚类的bboxes
    :param k:簇数(聚成几类)
    :param dist:更新簇坐标的方法(默认使用中位数，比均值效果略好)
    :return:
    """

    box_number = boxes.shape[0]
    last_nearest = np.zeros((box_number,))
    # np.random.seed(0)  # 固定随机数种子

    # init k clusters
    clusters = boxes[np.random.choice(box_number, k, replace=False)]

    while True:
        distances = 1 - wh_iou(boxes, clusters)
        current_nearest = np.argmin(distances, axis=1)
        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        for cluster in range(k):
            # update clusters
            clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)

        last_nearest = current_nearest

    return clusters
