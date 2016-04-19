import numpy as np
from PIL import Image
import os
from utils import *

TEST_TO_TRAIN_RATIO = 0.9
CHARS_PATH = './chars74k-lite/chars74k-lite/'
N_CLASSES = 26


def convert_y(y, n_classes):
    """
    Convert a list of numbers in the range [0,n_classes-1] to a list of lists each with length n_classes
    in the one-hot format.
    E.g. ([3], 10) --> [[0,0,0,1,0,0,0,0,0,0]]
    """
    y = y.flatten()
    arr = np.zeros((len(y), n_classes))
    arr[np.arange(len(y)), y] = 1
    return arr


def img_to_list(img_path, nbits=8):
    normalized_floats = np.asarray(Image.open(img_path)).astype(float) / (2**nbits-1)
    return normalized_floats.flatten()


def list_to_img(img, nbits=8):
    img_mul = img * (2**nbits-1)
    img_ints = np.rint(img_mul)
    return img_ints.astype(int)


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def load():
    # todo only load a percent of data set
    folders = os.listdir(CHARS_PATH)
    assert len(folders) == 26

    x = []
    y = []
    for num, alpha in enumerate(folders):
        folder = os.listdir(CHARS_PATH + alpha)
        for filename in folder:
            path = CHARS_PATH + alpha + "/" + filename
            x.append(img_to_list(path))
            y.append(num)

    x_shuffled, y_shuffled = shuffle_in_unison(np.array(x), np.array(y))
    y_shuffled = convert_y(y_shuffled, N_CLASSES)
    split_index = int(len(x_shuffled)*TEST_TO_TRAIN_RATIO)
    train_x = x_shuffled[:split_index]
    train_y = y_shuffled[:split_index]
    test_x = x_shuffled[split_index:]
    test_y = y_shuffled[split_index:]

    return DataSet(train_x, train_y, test_x, test_y)

