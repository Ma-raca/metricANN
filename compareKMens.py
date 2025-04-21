import numpy as np
import time
from sklearn.cluster import KMeans as SKLearnKMeans
from index_build.myClusterAlgorithm import KMeans as CustomKMeans
from fileIO.vectorIO import load_vec
import matplotlib.pyplot as plt
import os


def compare_kmeans(dataset, n_clusters, max_iter=100):
    """
    比较自定义KMeans与scikit-learn KMeans的性能

    参数:
        dataset: 数据集
        n_clusters: 聚类数量
        max_iter: 最大迭代次数

    返回:
        custom_time: 自定义KMeans的运行时间
        sklearn_time: scikit-learn KMeans的运行时间
    """
    print(f"数据集形状: {dataset.shape}")
    print(f"聚类数量: {n_clusters}")

    # 测试自定义KMeans
    print("\n运行自定义KMeans...")
    custom_kmeans = CustomKMeans(n_clusters=n_clusters, random_state=42, T=max_iter)
    start_time = time.time()
    custom_kmeans.fit(dataset)
    end_time = time.time()
    custom_time = end_time - start_time
    print(f"自定义KMeans用时: {custom_time:.2f}秒")

    # 测试scikit-learn KMeans
    print("\n运行scikit-learn KMeans...")
    sklearn_kmeans = SKLearnKMeans(n_clusters=n_clusters, random_state=42, max_iter=max_iter,
                                   init='random', n_init=1)  # 使用random初始化以匹配自定义实现
    start_time = time.time()
    sklearn_kmeans.fit(dataset)
    end_time = time.time()
    sklearn_time = end_time - start_time
    print(f"scikit-learn KMeans用时: {sklearn_time:.2f}秒")

    # 比较速度
    speed_ratio = custom_time / sklearn_time
    print(f"\n速度比较: 自定义实现比scikit-learn慢 {speed_ratio:.2f} 倍")

    # 比较聚类质量 - 计算惯性(inertia): 每个点到其聚类中心的平方距离之和
    custom_inertia = 0
    for i in range(dataset.shape[0]):
        center = custom_kmeans.centroids[custom_kmeans.labels[i]]
        custom_inertia += np.sum((dataset[i] - center) ** 2)

    sklearn_inertia = sklearn_kmeans.inertia_

    print(f"自定义KMeans惯性: {custom_inertia:.2f}")
    print(f"scikit-learn KMeans惯性: {sklearn_inertia:.2f}")
    print(f"惯性比率: {custom_inertia / sklearn_inertia:.4f}")

    return custom_time, sklearn_time, custom_inertia, sklearn_inertia


def compare_with_different_sizes(filepath, n_clusters=256, sizes=[1000, 5000, 10000, 50000, 100000]):
    """
    使用不同大小的数据子集进行比较
    """
    # 加载数据
    print(f"加载数据: {filepath}")
    dataset, dim, total_size = load_vec(filepath)
    print(f"数据集维度: {dim}, 总大小: {total_size}")

    # 确保所请求的最大大小不超过总数据量
    max_size = min(max(sizes), total_size)
    if max_size < max(sizes):
        sizes = [s for s in sizes if s <= max_size]
        print(f"调整大小列表为: {sizes} (受限于总数据量 {total_size})")

    results = []

    for size in sizes:
        print(f"\n---------- 测试大小: {size} ----------")
        data_subset = dataset[:size]
        custom_time, sklearn_time, custom_inertia, sklearn_inertia = compare_kmeans(data_subset, n_clusters)
        results.append((size, custom_time, sklearn_time, custom_inertia, sklearn_inertia))

    return results


def plot_results(results, save_path="kmeans_comparison_results.png"):
    """
    绘制比较结果
    """
    sizes = [r[0] for r in results]
    custom_times = [r[1] for r in results]
    sklearn_times = [r[2] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 时间比较图
    ax1.plot(sizes, custom_times, 'o-', label='自定义 KMeans', linewidth=2)
    ax1.plot(sizes, sklearn_times, 's-', label='scikit-learn KMeans', linewidth=2)
    ax1.set_title('KMeans 运行时间比较', fontsize=14)
    ax1.set_xlabel('数据集大小', fontsize=12)
    ax1.set_ylabel('运行时间 (秒)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)

    # 添加速度比率标签
    for i, (size, ct, st, _, _) in enumerate(results):
        ratio = ct / st
        ax1.annotate(f"{ratio:.2f}x",
                     xy=(size, max(ct, st)),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center', va='bottom',
                     fontsize=9)

    # 计算惯性比率
    inertia_ratios = [r[3] / r[4] for r in results]

    # 惯性比较图
    ax2.plot(sizes, inertia_ratios, 'o-', color='purple', linewidth=2)
    ax2.set_title('聚类质量比较 (惯性比率)', fontsize=14)
    ax2.set_xlabel('数据集大小', fontsize=12)
    ax2.set_ylabel('自定义/scikit-learn 惯性比率', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)  # 添加y=1的参考线

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"结果已保存为 '{save_path}'")


def compare_with_different_clusters(filepath, data_size=10000, cluster_list=[16, 64, 128, 256, 512]):
    """
    使用不同聚类数量进行比较
    """
    # 加载数据
    print(f"加载数据: {filepath}")
    dataset, dim, total_size = load_vec(filepath)
    print(f"数据集维度: {dim}, 总大小: {total_size}")

    # 使用指定大小的数据子集
    actual_size = min(data_size, total_size)
    data_subset = dataset[:actual_size]
    print(f"使用前 {actual_size} 个数据点")

    results = []

    for n_clusters in cluster_list:
        print(f"\n---------- 测试聚类数量: {n_clusters} ----------")
        custom_time, sklearn_time, custom_inertia, sklearn_inertia = compare_kmeans(data_subset, n_clusters)
        results.append((n_clusters, custom_time, sklearn_time, custom_inertia, sklearn_inertia))

    return results


def plot_cluster_results(results, save_path="kmeans_cluster_comparison.png"):
    """
    绘制不同聚类数量的比较结果
    """
    clusters = [r[0] for r in results]
    custom_times = [r[1] for r in results]
    sklearn_times = [r[2] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 时间比较图
    ax1.plot(clusters, custom_times, 'o-', label='自定义 KMeans', linewidth=2)
    ax1.plot(clusters, sklearn_times, 's-', label='scikit-learn KMeans', linewidth=2)
    ax1.set_title('不同聚类数量的运行时间比较', fontsize=14)
    ax1.set_xlabel('聚类数量', fontsize=12)
    ax1.set_ylabel('运行时间 (秒)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)

    # 添加速度比率标签
    for i, (clust, ct, st, _, _) in enumerate(results):
        ratio = ct / st
        ax1.annotate(f"{ratio:.2f}x",
                     xy=(clust, max(ct, st)),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center', va='bottom',
                     fontsize=9)

    # 计算惯性比率
    inertia_ratios = [r[3] / r[4] for r in results]

    # 惯性比较图
    ax2.plot(clusters, inertia_ratios, 'o-', color='purple', linewidth=2)
    ax2.set_title('不同聚类数量的质量比较 (惯性比率)', fontsize=14)
    ax2.set_xlabel('聚类数量', fontsize=12)
    ax2.set_ylabel('自定义/scikit-learn 惯性比率', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)  # 添加y=1的参考线

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"结果已保存为 '{save_path}'")


if __name__ == "__main__":
    # 您的数据路径
    filepath = "D:\\python\\metricANN\\data\\UniformVector20d\\randomvector20d1m.txt"

    # 比较不同数据集大小
    sizes = [1000, 5000, 10000, 50000, 100000]
    if os.path.exists(filepath):
        print("=== 比较不同数据集大小 ===")
        results_sizes = compare_with_different_sizes(filepath, n_clusters=256, sizes=sizes)
        plot_results(results_sizes)

        # 比较不同聚类数量
        print("\n\n=== 比较不同聚类数量 ===")
        cluster_list = [16, 64, 128, 256, 512]
        results_clusters = compare_with_different_clusters(filepath, data_size=10000, cluster_list=cluster_list)
        plot_cluster_results(results_clusters)
    else:
        print(f"错误: 找不到数据文件 {filepath}")
        print("请确认文件路径是否正确")