import numpy as np


def minkowski_distance(x, y, p=2, vectorized=False):
    """
    计算两个向量之间的闵可夫斯基距离
    
    参数:
        x: 向量或矩阵，如果是矩阵，形状为 (n_samples, n_features)
        y: 向量或矩阵，如果是矩阵，形状为 (m_samples, n_features)
           如果vectorized=True且y是向量，则将广播计算x中每个样本到y的距离
        p: 闵可夫斯基距离的参数，默认为2（欧氏距离）
           p=1: 曼哈顿距离
           p=2: 欧氏距离
           p=np.inf: 切比雪夫距离
        vectorized: 是否使用向量化计算，默认为True
        
    返回:
        如果x和y都是单个向量: 返回一个标量，表示它们之间的距离
        如果x是矩阵，y是向量且vectorized=True: 返回一个向量，表示x中每个样本到y的距离
        如果x和y都是矩阵: 返回距离矩阵，形状为(n_samples, m_samples)
    """
    # 将输入转换为NumPy数组
    x = np.asarray(x)
    y = np.asarray(y)
    
    # 非向量化计算模式
    if not vectorized:
        # 判断是否为单向量计算
        if x.ndim == 1 and y.ndim == 1:
            # 确保维度匹配
            if x.shape != y.shape:
                raise ValueError(f"向量维度不匹配: x的维度是{x.shape}，y的维度是{y.shape}")
                
            # 计算单向量距离
            if p == np.inf:  # 切比雪夫距离
                return np.max(np.abs(x - y))
            else:  # 其他闵可夫斯基距离
                return (np.sum((np.abs(x - y))**p))**(1/p)
        else:
            # 批量距离计算（非向量化）
            # 确保y是二维数组
            if y.ndim == 1:
                y = y.reshape(1, -1)
            if x.ndim == 1:
                x = x.reshape(1, -1)
                
            # 检查特征维度是否匹配
            if x.shape[1] != y.shape[1]:
                raise ValueError(f"特征维度不匹配: x的特征维度是{x.shape[1]}，y的特征维度是{y.shape[1]}")
                
            # 创建结果矩阵
            distances = np.zeros((x.shape[0], y.shape[0]))
            
            # 逐点计算距离
            for i in range(x.shape[0]):
                for j in range(y.shape[0]):
                    if p == np.inf:
                        distances[i, j] = np.max(np.abs(x[i] - y[j]))
                    else:
                        distances[i, j] = (np.sum((np.abs(x[i] - y[j]))**p))**(1/p)
                        
            return distances
    # 向量化计算模式
    else:
        # 单向量对单向量
        if x.ndim == 1 and y.ndim == 1:
            if x.shape != y.shape:
                raise ValueError(f"向量维度不匹配: x的维度是{x.shape}，y的维度是{y.shape}")
                
            if p == np.inf:
                return np.max(np.abs(x - y))
            else:
                return (np.sum((np.abs(x - y))**p))**(1/p)
                
        # 矩阵对向量（每行对单个向量）
        elif x.ndim == 2 and y.ndim == 1:
            if x.shape[1] != y.shape[0]:
                raise ValueError(f"特征维度不匹配: x的特征维度是{x.shape[1]}，y的维度是{y.shape[0]}")
                
            if p == np.inf:
                return np.max(np.abs(x - y), axis=1)
            else:
                return np.power(np.sum(np.power(np.abs(x - y), p), axis=1), 1/p)
                
        # 矩阵对矩阵（批量计算两组向量间的距离）
        else:
            # 确保输入是二维
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if y.ndim == 1:
                y = y.reshape(1, -1)
                
            if x.shape[1] != y.shape[1]:
                raise ValueError(f"特征维度不匹配: x的特征维度是{x.shape[1]}，y的特征维度是{y.shape[1]}")
            
            # 矩阵对矩阵的距离计算需要用到广播
            # 扩展维度: x形状(n,1,features), y形状(1,m,features)
            x_expanded = x[:, np.newaxis, :]
            
            if p == np.inf:
                return np.max(np.abs(x_expanded - y), axis=2)
            else:
                return np.power(np.sum(np.power(np.abs(x_expanded - y), p), axis=2), 1/p)


# 使用示例
if __name__ == "__main__":
    # 创建两个示例向量
    vec1 = np.array([1, 2, 3, 4, 5])
    vec2 = np.array([5, 4, 3, 2, 1])

    # 创建多个向量进行批量测试
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    Y = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

    print("单点距离计算:")
    # 向量化模式
    manhattan_vec = minkowski_distance(vec1, vec2, p=1)
    euclidean_vec = minkowski_distance(vec1, vec2, p=2)
    chebyshev_vec = minkowski_distance(vec1, vec2, p=np.inf)

    # 非向量化模式
    manhattan_nonvec = minkowski_distance(vec1, vec2, p=1, vectorized=False)
    euclidean_nonvec = minkowski_distance(vec1, vec2, p=2, vectorized=False)
    chebyshev_nonvec = minkowski_distance(vec1, vec2, p=np.inf, vectorized=False)

    print(f"曼哈顿距离 (p=1): {manhattan_vec}")
    print(f"欧氏距离 (p=2): {euclidean_vec}")
    print(f"切比雪夫距离 (p=∞): {chebyshev_vec}")

    print("\n验证向量化与非向量化结果一致性:")
    print(f"曼哈顿距离一致: {manhattan_vec == manhattan_nonvec}")
    print(f"欧氏距离一致: {euclidean_vec == euclidean_nonvec}")
    print(f"切比雪夫距离一致: {chebyshev_vec == chebyshev_nonvec}")

    print("\n批量距离计算:")
    # 矩阵对矩阵距离
    matrix_dist_vec = minkowski_distance(X, Y, p=2)
    matrix_dist_nonvec = minkowski_distance(X, Y, p=2, vectorized=False)

    print("向量化结果:")
    print(matrix_dist_vec)
    print("\n非向量化结果:")
    print(matrix_dist_nonvec)
    print(f"结果一致: {np.allclose(matrix_dist_vec, matrix_dist_nonvec)}")

    # 矩阵对向量距离
    print("\n矩阵对向量距离计算:")
    matrix_vec_dist = minkowski_distance(X, Y[0], p=2)
    print(matrix_vec_dist)

    # 性能比较
    import time

    # 创建较大的测试数据
    n_samples = 1000
    n_features = 10
    X_large = np.random.random((n_samples, n_features))
    Y_large = np.random.random((n_samples, n_features))

    print("\n性能测试:")

    # 向量化
    start = time.time()
    _ = minkowski_distance(X_large, Y_large[0], p=2)
    vec_time = time.time() - start
    print(f"向量化计算 {n_samples} 个距离用时: {vec_time:.6f} 秒")

    # 非向量化
    start = time.time()
    _ = minkowski_distance(X_large, Y_large[0], p=2, vectorized=False)
    nonvec_time = time.time() - start
    print(f"非向量化计算 {n_samples} 个距离用时: {nonvec_time:.6f} 秒")
    print(f"性能提升: {nonvec_time / vec_time:.2f}x")


