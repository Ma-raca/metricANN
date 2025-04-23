from metricDist.minkowskiDist import minkowski_distance
from metricDist.editDist import edit_distance
import numpy as np
from fileIO.vectorIO import load_vec
from pivotSpace.pivotSelect import farthest_first_traversal
import time


def linear_scan_vec(dataset, query, r, k, search='range', p=2):
    """
    使用向量化距离计算实现的线性扫描算法

    参数:
        dataset: 数据集，NumPy数组
        query: 查询点，NumPy数组
        r: 范围查询的半径
        k: KNN查询返回的邻居数量
        search: 查询类型，'range'或'knn'
        p: 闵可夫斯基距离的参数，默认为2（欧几里得距离）

    返回:
        查询结果
    """

    # 使用向量化计算所有距离

    # range query
    if search == 'range':
        # 找出距离小于r的点
        dist = minkowski_distance(dataset, query, p=2)
        matching_indices = np.where(dist < r)[0]
        results = dataset[matching_indices]
    else:
        # Nearest Neighbor query
        dist = minkowski_distance(dataset, query, p=2)
        sorted_indices = np.argsort(dist)
        nearest_indices = sorted_indices[:k]
        results = dataset[nearest_indices]

    return results


def linear_scan(dataset, query, r, k, search='range', p=2):
    """
    使用逐点距离计算实现的线性扫描算法
    
    参数:
        dataset: 数据集，NumPy数组
        query: 查询点，NumPy数组
        r: 范围查询的半径
        k: KNN查询返回的邻居数量
        search: 查询类型，'range'或'knn'
        p: 闵可夫斯基距离的参数，默认为2（欧几里得距离）
        
    返回:
        查询结果
    """
    start_time = time.time()  # 开始计时
    n_samples = dataset.shape[0]

    # range query
    if search == 'range':
        # 找出距离小于r的点
        matching_indices = []

        for i in range(n_samples):
            # 逐点计算距离
            dist = minkowski_distance(dataset[i], query, p=p, vectorized=False)
            if dist < r:
                matching_indices.append(i)

        results = dataset[matching_indices]
    else:
        # Nearest Neighbor query
        # 创建数组存储所有距离
        distances = np.zeros(n_samples)

        for i in range(n_samples):
            # 逐点计算距离
            distances[i] = minkowski_distance(dataset[i], query, p=p, vectorized=False)

        # 找出k个最近的点
        sorted_indices = np.argsort(distances)
        nearest_indices = sorted_indices[:k]
        results = dataset[nearest_indices]
        
    end_time = time.time()  # 结束计时
    print(f"线性扫描用时: {end_time - start_time:.4f}秒")
    
    return results


if __name__ == "__main__":
    filepath = "D:\\python\\metricANN\data\\texas\\texas.txt"
    dataset, dim, size = load_vec(filepath)
    query = np.array([23, 132])
    k = 5
    r = 170
    search = 'knn'
    results = linear_scan(dataset, query, r, k, search)
    print(results)
    selected_indices, selected_points = farthest_first_traversal(dataset, k)
    pivotSet = selected_points
    print(selected_indices)
