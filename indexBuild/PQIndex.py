from fileIO.vectorIO import load_vec
from metricDist.minkowskiDist import minkowski_distance
from pivotSpace.pivotSelect import farthest_first_traversal
import numpy as np
from search.linear_scan import linear_scan
from sklearn.cluster import KMeans


def ProductQuantization_Train(dataset, m, k):
    # 1.确保维度能被 m 整除
    n, d = dataset.shape
    if d % m != 0:
        # 计算需要补充的维度
        pad_dims = m - (d % m)
        # 创建填充数组
        padding = np.zeros((n, pad_dims))
        # 连接原始数据和填充
        dataset = np.hstack((dataset, padding))
        # 更新维度
        _, d = dataset.shape

    d_star = d // m

    # 2.将数据集分解为 m 个子空间
    subspaces = []
    for j in range(m):
        # 提取第 j 个子空间的数据
        start_dim = j * d_star
        end_dim = (j + 1) * d_star
        subspace_j = dataset[:, start_dim:end_dim]
        subspaces.append(subspace_j)

    # 3.在每个子空间上应用 K-means 聚类，得到codebooks
    # codebooks是聚类中心
    codebooks = []
    for j in range(m):
        print(f"在子空间 {j} 上进行 K-means 聚类...")
        # 使用scikit-learn的KMeans替代自定义实现
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=1)
        kmeans.fit(subspaces[j])
        codebooks.append(kmeans.cluster_centers_)
        print(f"子空间 {j} 的码本形状: {kmeans.cluster_centers_.shape}")

    # 4. 使用codebooks对数据进行编码
    codes = np.zeros((n, m), dtype=np.int32)
    for j in range(m):
        # 计算当前子空间中每个点到聚类中心的距离
        distances = np.zeros((n, k))
        for i in range(k):
            # 计算到第 i 个聚类中心的距离
            center = codebooks[j][i].reshape(1, -1)
            distances[:, i] = np.sum((subspaces[j] - center) ** 2, axis=1)

        # 找到最近的聚类中心
        # argmin相当于返回的是所属cluster的index，有256个聚类中心，也就是说数据的index是0-255，2的8次方，每个子空间的一个数据只用8bit,切成m份，就是m*8bit
        codes[:, j] = np.argmin(distances, axis=1)

    print(f"编码完成，编码形状: {codes.shape}")

    return codebooks, codes


if __name__ == '__main__':
    filepath = "D:\\python\\metricANN\\data\\UniformVector20d\\randomvector20d1m.txt"
    dataset, dim, size = load_vec(filepath)
    dataset = dataset[0:1000, :]
    query = np.array([-94.04345, 33.552254])

    # selected_indices, selected_points = farthest_first_traversal(dataset, k)
    # pivotSet = selected_points
    # print(selected_indices)
    k = 256
    m = 8
    ProductQuantization_Train(dataset, m, k)
