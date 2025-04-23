from indexBuild.PQIndex import ProductQuantization_Train
from fileIO.vectorIO import load_vec
from metricDist.minkowskiDist import minkowski_distance
from pivotSpace.pivotSelect import farthest_first_traversal
import numpy as np
import time


def ProductQuantization_Search(query, codebooks, codes, knnNum):
    """
    使用乘积量化进行最近邻搜索
    
    参数:
        query: 查询向量
        codebooks: PQ码本列表
        codes: 数据库向量的PQ编码
        knnNum: 返回的最近邻数量
        
    返回:
        nearest_indices: 最近的k个点的索引
        distances: 对应的近似距离
    """
    start_time = time.time()
    # 1.准备阶段
    query = np.expand_dims(query, axis=0)
    m = len(codebooks)
    n = len(codes)
    d_star = codebooks[0].shape[1]
    queryDim = query.shape[1]
    # 检查query的维度是否与codebooks一致
    if queryDim != m * d_star:
        print("Query dimension does not match codebook dimension.")
        if query.shape[1] < m * d_star:
            # 计算需要补充的维度
            pad_dims = m * d_star - queryDim
            # 创建填充数组
            padding = np.zeros((1, pad_dims))

            # 连接原始数据和填充
            query = np.hstack((query, padding))
            # 更新维度
            _, queryDim = query.shape
        else:
            # 截断query
            query = query[:, 0:m * d_star]

    # 2.构建距离表，将query分割成m个子空间，并计算子空间到码本的距离
    subspacesQuery = []
    distanceQuery = np.zeros((m, codebooks[0].shape[0]))
    # codeQuery = np.zeros((1, m))
    for j in range(m):
        # 提取第 j 个子空间的数据
        start_dim = j * d_star
        end_dim = (j + 1) * d_star
        subspace_j = query[:, start_dim:end_dim]
        subspacesQuery.append(subspace_j)
        # 计算第 j 个子空间到码本的距离，也就是第 j 个子空间到各个cluster的距离
        for i in range(codebooks[0].shape[0]):
            distanceQuery[j][i] = minkowski_distance(subspace_j, codebooks[j][i])

    # 3.累积距离计算
    # 遍历被编码后的数据，读取这些数据的所属cluster的index，再找到distanceQuery对应的距离值，再把这些距离值加起来。
    # 原理：PQ 假设总距离可以近似为各个子空间距离之和。
    approx_distances = np.zeros(n)
    for i in range(n):
        for j in range(m):
            code_idx = codes[i, j]
            approx_distances[i] += distanceQuery[j, code_idx]

    nearest_indices = np.argsort(approx_distances)[:knnNum]
    end_time = time.time()
    print(f"PQ搜索用时: {end_time - start_time:.4f}秒")
    return nearest_indices, approx_distances[nearest_indices]


if __name__ == '__main__':
    filepath = "D:\\python\\metricANN\\data\\UniformVector20d\\randomvector20d1m.txt"
    dataset, dim, size = load_vec(filepath)
    query = np.array(
        [-94.04345, 353.552254, -914.04345, 323.552254, -914.04345, 33.552254, -984.04345, 33.552254, -994.04345,
         43.55254, -974.04345, 353.552254, -4.04345, 37.552254, 323.552254, 353.552254, 33.552254, 33.552254, 3.552254,
         3.552254])

    # selected_indices, selected_points = farthest_first_traversal(dataset, k)
    # pivotSet = selected_points
    # print(selected_indices)
    k = 256
    m = 8
    knnNum = 5
    codebooks, codes = ProductQuantization_Train(dataset, m, k)
    results, distances = ProductQuantization_Search(query, codebooks, codes, knnNum)
    print(f"查询结果: {results}")
    print(f"对应距离: {distances}")
