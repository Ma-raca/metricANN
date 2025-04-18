import numpy as np


def edit_distance(word1, word2):
    pass
    n = len(word1)
    m = len(word2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
    editDist = dp[n][m]
    return editDist


def weight_edit_distance(word1, word2):
    n = len(word1)
    m = len(word2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
    editDist = dp[n][m]
    pass
    return editDist


# 使用示例
if __name__ == "__main__":
    editdist = edit_distance(word1='abcd', word2='esadasdasdfgh')

    # 创建两个示例向量
    vec1 = np.array([1, 2, 3, 4, 5])
    vec2 = np.array([5, 4, 3, 2, 1])

    # 计算不同p值的闵可夫斯基距离
    manhattan = minkowski_distance(vec1, vec2, p=1)  # 曼哈顿距离
    euclidean = minkowski_distance(vec1, vec2, p=2)  # 欧氏距离
    chebyshev = minkowski_distance(vec1, vec2, p=np.inf)  # 切比雪夫距
