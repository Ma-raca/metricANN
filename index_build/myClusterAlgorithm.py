import numpy as np
from metricDist.minkowskiDist import minkowski_distance
import time

class KMeans:
    def __init__(self, n_clusters, random_state=42, T=10, tol=1e-4):
        """
        KMeans 聚类算法

        参数:
            n_clusters: 聚类数量
            random_state: 随机种子
            T: 最大迭代次数
            tol: 收敛容差
        """
        self.labels = None
        self.centroids = None
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.T = T
        self.tol = tol

    def _assign_labels_vectorized(self, datas):
        """
        向量化方式为所有点分配标签
        """
        # 计算所有点到所有中心的距离矩阵 (n_samples, n_clusters)
        n_samples = datas.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))

        for j in range(self.n_clusters):
            # 使用广播机制计算每个点到当前中心的距离
            diff = datas - self.centroids[j]
            distances[:, j] = np.sum(diff * diff, axis=1)

        # 为每个点找到最近的中心点
        return np.argmin(distances, axis=1)

    def fit(self, datas):
        """
        训练KMeans模型

        参数:
            datas: 数据集，形状为(n_samples, n_features)
        返回:
            self
        """
        start_time = time.time()
        # 随机选择初始聚类中心
        np.random.seed(self.random_state)
        n_samples = datas.shape[0]
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = datas[random_indices].copy()
        self.labels = np.zeros(n_samples, dtype=np.int32)

        # 迭代优化
        for t in range(1, self.T + 1):
            # 向量化方式分配标签
            new_labels = self._assign_labels_vectorized(datas)

            # 如果是第二次及以后的迭代，检查收敛
            if t > 1 and np.array_equal(new_labels, self.labels):
                print(f"KMeans收敛于迭代 {t}/{self.T}")
                break

            self.labels = new_labels

            # 保存旧中心点以检查移动距离
            old_centroids = self.centroids.copy()

            # 更新聚类中心
            empty_clusters = []
            for i in range(self.n_clusters):
                cluster_points = datas[self.labels == i]
                if len(cluster_points) > 0:
                    self.centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    empty_clusters.append(i)

            # 处理空聚类
            if empty_clusters:
                print(f"警告: 迭代 {t} 中发现 {len(empty_clusters)} 个空聚类")
                for i in empty_clusters:
                    # 找到样本中距离最远的点作为新中心
                    farthest_idx = np.random.randint(0, n_samples)
                    self.centroids[i] = datas[farthest_idx].copy()

            # 计算中心点移动距离
            center_shift = np.linalg.norm(self.centroids - old_centroids)
            if center_shift < self.tol:
                print(f"KMeans中心点收敛于迭代 {t}/{self.T}，移动距离: {center_shift:.6f}")
                break

        end_time = time.time()
        print(f"KMeans训练完成，用时: {end_time - start_time:.2f}秒")
        return self


# 如果直接运行此文件，执行测试代码
if __name__ == "__main__":
    # 生成随机测试数据
    np.random.seed(42)
    test_data = np.random.rand(1000, 2) * 10

    # 创建KMeans实例并训练
    kmeans = KMeans(n_clusters=5, T=20)
    kmeans.fit(test_data)

    # 输出聚类结果
    print("聚类中心:")
    print(kmeans.centroids)

    # 可视化聚类结果
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.scatter(test_data[:, 0], test_data[:, 1], c=kmeans.labels, cmap='viridis', alpha=0.7)
        plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='*', s=200, label='聚类中心')
        plt.title('KMeans聚类结果')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('kmeans_test.png')
        plt.show()
        print("聚类结果已保存为 'kmeans_test.png'")
    except ImportError:
        print("未安装matplotlib，无法可视化聚类结果")