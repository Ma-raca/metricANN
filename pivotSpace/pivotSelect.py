from metricDist.minkowskiDist import minkowski_distance
from metricDist.editDist import edit_distance
from fileIO.vectorIO import load_vec
import numpy as np
import matplotlib.pyplot as plt


def farthest_first_traversal(data, k, metric='euclidean', initial_idx=None, random_state=None):
    """
    实现最远优先遍历算法，从数据集中选择k个最具代表性的点

    参数:
        data (np.ndarray): 形状为 (n_samples, n_features) 的数据集
        k (int): 要选择的点数量
        metric (str or callable): 距离度量，默认为欧氏距离
        initial_idx (int): 初始点的索引，如果为None则随机选择
        random_state (int): 随机数生成器的种子

    返回:
        tuple: (选择的点的索引, 选择的点的数据)
    """
    n_samples = data.shape[0]

    # 检查参数有效性
    if k <= 0 or k > n_samples:
        raise ValueError(f"参数k必须在1和数据集大小({n_samples})之间")

    # 设置随机种子
    if random_state is not None:
        np.random.seed(random_state)

    # 初始化距离矩阵
    distances = np.zeros(n_samples)

    # 已选择的点的索引
    selected_indices = []

    # 选择初始点
    if initial_idx is None:
        initial_idx = np.random.randint(0, n_samples)

    selected_indices.append(initial_idx)

    # 计算到初始点的距离
    for i in range(n_samples):
        distances[i] = minkowski_distance(data[i], data[initial_idx],2,0)

    # 选择剩余的k-1个点
    for _ in range(1, k):
        # 找到距离已选点最远的点
        next_idx = np.argmax(distances)
        selected_indices.append(next_idx)

        # 更新距离：使用最小距离
        for i in range(n_samples):
            new_dist = minkowski_distance(data[i], data[next_idx],2,0)
            distances[i] = min(distances[i], new_dist)

    # 提取选择的点
    selected_points = data[selected_indices]

    return selected_indices, selected_points


if __name__ == "__main__":
    # 加载数据集
    file_path = "D:\\python\\metricANN\\data\\texas\\texas.txt"  # 替换为您的数据集路径
    dataset, dim, size = load_vec(file_path)

    # 选择点数
    num_points = 3

    # 使用FFT选择代表性点
    indices, selected_points = farthest_first_traversal(dataset, num_points, random_state=42)

    print(f"选择了{num_points}个代表性点")
    print(f"选择的点的索引: {indices}")

    # 如果数据是2D的，可视化结果
    if dim == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(dataset[:, 0], dataset[:, 1], c='lightgray', alpha=0.5, label='原始数据点')
        plt.scatter(selected_points[:, 0], selected_points[:, 1], c='red', s=100, label='FFT选择的点')

        # 标记选择点的顺序
        for i, point in enumerate(selected_points):
            plt.annotate(str(i + 1), (point[0], point[1]), xytext=(5, 5), textcoords='offset points')

        plt.legend()
        plt.title('最远优先遍历算法选择结果')
        plt.show()