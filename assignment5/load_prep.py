import numpy as np
from PIL import Image
import os
from utils import *
import logging

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


def load_img(img_path, img_mode='L'):
    try:
        img = Image.open(img_path).convert(img_mode)
        return img
    except AttributeError as e:
        logging.error("Error: %s \n Path: %s", e, img_path)


def img_to_list(img, flatten, nbits=8, img_mode='L'):
    img = img.convert(img_mode)
    img_bytes = img.tobytes("raw", img_mode)
    ints = np.fromstring(img_bytes, dtype='uint8')
    scaled_floats = ints.astype(dtype='float32') / (2**nbits - 1)
    if flatten:
        return scaled_floats.flatten()
    else:
        return scaled_floats.reshape(img.size)


def list_to_img(img_as_list, img_size=None, nbits=8, img_mode='L'):
    assert type(img_as_list) == np.ndarray, type(img_as_list)
    if img_size is None:
        assert img_as_list.ndim == 2, ("Can't handle flattened arrays if not passed an img size", img_as_list.shape)
        img_size = img_as_list.shape
    scaled = img_as_list * (2**nbits - 1)
    rounded = np.rint(scaled)
    ints = rounded.astype(dtype='uint8')
    img = Image.new(img_mode, img_size)
    img.putdata(ints.flatten())
    return img


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def invert_image(img_list):
    return 1 - img_list


def load(prep_funcs=None):
    folders = os.listdir(CHARS_PATH)
    assert len(folders) == 26

    x = []
    y = []
    for num, alpha in enumerate(folders):
        folder = os.listdir(CHARS_PATH + alpha)
        for filename in folder:
            path = CHARS_PATH + alpha + "/" + filename
            x.append(load_img(path))
            y.append(num)
    logging.info("\tLoaded %s examples. Splitting into testing and training data sets with a ratio of %s",
                 len(x), TEST_TO_TRAIN_RATIO)

    x_shuffled, y_shuffled = shuffle_in_unison(np.asarray(x, dtype=object), np.asarray(y))
    y_shuffled = convert_y(y_shuffled, N_CLASSES)
    split_index = int(len(x_shuffled)*TEST_TO_TRAIN_RATIO)
    train_x = x_shuffled[:split_index]
    train_y = y_shuffled[:split_index]
    test_x = x_shuffled[split_index:]
    test_y = y_shuffled[split_index:]

    train_x_prepped = []
    # The y-values (labels) themselves are not prepped/changed in any way, but since there are more training examples
    # there has to be more labels.
    train_y_prepped = []
    for prep_func in prep_funcs:
        train_x_prepped.extend(map(prep_func, train_x))
        train_y_prepped.extend(train_y)
    if prep_funcs:
        train_x_prepped.extend(train_x)
        train_y_prepped.extend(train_y)
        logging.debug('\tExpanded training data set from %s examples to %s examples', len(train_x), len(train_x_prepped))
        train_x, train_y = shuffle_in_unison(np.asarray(train_x_prepped, dtype=object), np.asarray(train_y_prepped))

    train_x = np.asarray([img_to_list(img, flatten=True) for img in train_x])
    test_x = np.asarray([img_to_list(img, flatten=True) for img in test_x])
    #print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    #print(train_x, train_y, test_x, test_y)
    return DataSet(train_x, train_y, test_x, test_y)

