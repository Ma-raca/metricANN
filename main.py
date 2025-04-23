from fileIO.vectorIO import load_vec
from metricDist.minkowskiDist import minkowski_distance
from pivotSpace.pivotSelect import farthest_first_traversal
import numpy as np
from search.linear_scan import linear_scan
from indexBuild.PQIndex import ProductQuantization_Train
from search.PQsearch import ProductQuantization_Search


def main(filepath):

    # 数据准备
    dataset, dim, size = load_vec(filepath)
    query = np.array(
        [-94.04345, 353.552254, -914.04345, 323.552254, -914.04345, 33.552254, -984.04345, 33.552254, -994.04345,
         43.55254, -974.04345, 353.552254, -4.04345, 37.552254, 323.552254, 353.552254, 33.552254, 33.552254, 3.552254,
         3.552254])
    knnNum = 100
    rangeR = 5

    # 线性扫描
    search = 'knn'
    results = linear_scan(dataset, query, rangeR, knnNum, search)
    print(results)

    # 支撑点空间建立
    # selected_indices, selected_points = farthest_first_traversal(dataset, k)
    # pivotSet = selected_points
    # print(selected_indices)

    # PQ查询
    subspaceNum = 8  # 子空间个数
    KMeansNum = 256  # 子空间聚类个数
    codebooks, codes = ProductQuantization_Train(dataset, subspaceNum, KMeansNum)
    results, distances = ProductQuantization_Search(query, codebooks, codes, knnNum)
    print(f"查询结果: {results}")
    print(f"对应距离: {distances}")


if __name__ == "__main__":
    filepath = "D:\\python\\metricANN\\data\\randomvector5d1m.txt"
    main(filepath)
