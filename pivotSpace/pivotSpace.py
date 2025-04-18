from metricDist.minkowskiDist import minkowski_distance
from metricDist.editDist import edit_distance
from fileIO.vectorIO import load_vec
import numpy as np
import matplotlib.pyplot as plt
from pivotSelect import farthest_first_traversal


def pivot_space_vectorized(dataset, pivotSet, k):
    # 使用批量距离计算
    distMatrix = np.zeros((dataset.shape[0], k))
    
    for j in range(k):
        # 向量化计算所有数据点到当前pivot点的距离
        distMatrix[:, j] = minkowski_distance(dataset, pivotSet[j], p=2, vectorized=True)
    
    return distMatrix


def pivot_space(dataset, pivotSet, k):
    # 正确创建 n 行 k 列的全零数组
    distMatrix = np.zeros((dataset.shape[0], k))
    
    for i in range(dataset.shape[0]):
        for j in range(k):
            distMatrix[i, j] = minkowski_distance(dataset[i], pivotSet[j], 1)
    
    return distMatrix


if __name__ == "__main__":
    # 加载数据集
    file_path = "D:\\python\\metricANN\\data\\texas\\texas.txt"  # 替换为您的数据集路径
    dataset, dim, size = load_vec(file_path)

    # 选择点数
    num_points = 3

    # 使用FFT选择代表性点
    indices, selected_points = farthest_first_traversal(dataset, num_points, random_state=42)
    pivotSpace = pivot_space(dataset, selected_points, num_points)

    print(f"选择了{num_points}个代表性点")
    print(f"选择的点的索引: {indices}")


