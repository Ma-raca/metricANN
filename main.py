from fileIO.vectorIO import load_vec
from metricDist.minkowskiDist import minkowski_distance
from pivotSpace.pivotSelect import farthest_first_traversal
import numpy as np
from search.linear_scan import linear_scan


def main(filepath):

    dataset, dim, size = load_vec(filepath)
    query = np.array([-94.04345, 33.552254])
    k = 100
    r = 5
    search = 'knn'
    results = linear_scan(dataset, query, r, k, search)
    print(results)
    # selected_indices, selected_points = farthest_first_traversal(dataset, k)
    # pivotSet = selected_points
    # print(selected_indices)


if __name__ == "__main__":
    filepath = "D:\\python\\metricANN\\data\\randomvector5d1m.txt"
    main(filepath)
