from fileIO.vectorIO import load_vec
from metricDist.minkowskiDist import minkowski_distance
from pivotSpace.pivotSelect import farthest_first_traversal
import numpy as np
from search.linear_scan import linear_scan


def ProductQuantization_Train(dataset, m, k):
    pass



if __name__ == '__main__':
    filepath = "D:\\python\\metricANN\data\\texas\\texas.txt"
    dataset, dim, size = load_vec(filepath)
    query = np.array([-94.04345, 33.552254])

    # selected_indices, selected_points = farthest_first_traversal(dataset, k)
    # pivotSet = selected_points
    # print(selected_indices)
    m=5
    k=5
    ProductQuantization_Train(dataset, m, k)
