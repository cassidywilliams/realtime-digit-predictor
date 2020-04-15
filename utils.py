import numpy as np
from skimage.measure import block_reduce


def list_to_sparse_matrix(coords, size=280):
    arr = np.zeros((size, size))
    for point in coords:
        arr[point[1], point[0]] = 1

    return arr.astype(int)


def resize_img(input_array, size=10):
    reshaped = block_reduce(input_array, (size, size), np.max)
    return reshaped.reshape(1, 28, 28, 1)