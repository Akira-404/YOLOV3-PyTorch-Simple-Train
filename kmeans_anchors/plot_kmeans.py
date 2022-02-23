import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)

# color array
colors = np.array(['blue', 'black'])


def plot_clusters(data: np.ndarray, cls, clusters: np.ndarray, title: str = "") -> None:
    if cls is None:
        c = [colors[0]] * data.shape[0]
    else:
        c = colors[cls].tolist()

    plt.scatter(data[:, 0], data[:, 1], c=c)
    for i, clus in enumerate(clusters):
        plt.scatter(clus[0], clus[1], c='gold', marker='*', s=150)
    plt.title(title)
    plt.show()
    plt.close()


# SSE:SUM OF SQUARED ERROR
def sum_squared_error(data: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    xy1 = data[:, None]  # [N,2]->[N,1,2]
    xy2 = clusters[None]  # [M,2]->[1,M,2]
    # xy1 and xy2 shape:[N,M,2]
    squared_error = np.power(xy2 - xy1, 2)
    see = np.sum(squared_error, axis=-1)  # axis=-1==axis=2
    return see


def k_means(data: np.ndarray, k: int, function=np.mean) -> np.ndarray:
    """
    :param data:需要聚类的data
    :param k:簇数(聚成几类)
    :param function:更新簇坐标的方法
    :return:
    """

    data_number = data.shape[0]
    print(f'data number:{data_number}')
    last_nearest = np.zeros((data_number,))

    # init k clusters 簇心 shape:[[x1,y1],[x2,y2]]
    clusters = data[np.random.choice(data_number, k, replace=False)]
    print(f"random cluster: \n {clusters}")
    # plot
    plot_clusters(data, None, clusters, "random clusters")

    step = 0
    while True:
        d = sum_squared_error(data, clusters)
        current_nearest = np.argmin(d, axis=1)  # 对所有数据进行聚类中心划分

        # plot
        plot_clusters(data, current_nearest, clusters, f"step {step}")

        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        for cluster in range(k):
            # update clusters
            clusters[cluster] = function(data[current_nearest == cluster], axis=0)
        last_nearest = current_nearest
        step += 1

    return clusters


def main():
    x1, y1 = [np.random.normal(loc=1., size=150) for _ in range(2)]
    x2, y2 = [np.random.normal(loc=5., size=150) for _ in range(2)]

    x = np.concatenate([x1, x2])  # shape:[300,]
    y = np.concatenate([y1, y2])  # shape:[300,]

    plt.scatter(x, y, c='blue')
    plt.title("initial data")
    plt.show()
    plt.close()

    # x[:, None]:(300,) -> (300,1)
    # y[:, None]:(300,) -> (300,1)
    # data shape:[300,2]
    data = np.concatenate([x[:, None], y[:, None]], axis=-1)  # axis=-1在最后一维操作
    clusters = k_means(data, k=2)
    print(f"k-means cluster: \n {clusters}")


if __name__ == '__main__':
    main()
