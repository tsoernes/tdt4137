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
    assert np.max(y) < n_classes, np.min(y) > 0
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


def load_img(img_path):
    """
    :param img_path:
    :return: PIL image
    """
    try:
        img = Image.open(img_path).convert('L')
        return img
    except AttributeError as e:
        logging.error("Error: %s \n Path: %s", e, img_path)
        raise


def load_images():
    folders = os.listdir(CHARS_PATH)
    assert len(folders) == N_CLASSES
    x = []
    y = []
    for num, alpha in enumerate(folders):
        folder = os.listdir(CHARS_PATH + alpha)
        for filename in folder:
            path = CHARS_PATH + alpha + "/" + filename
            x.append(load_img(path))
            y.append(num)
    return x, y


def img_to_list(img, flatten):
    """
    Can only handle 8-bit gray-scale
    :param img:
    :param flatten:
    :return: numpy.ndarray
    """
    img = img.convert('L')
    img_bytes = img.tobytes("raw", 'L')
    ints = np.fromstring(img_bytes, dtype='uint8')
    scaled_floats = ints.astype(dtype='float32') / (2**8 - 1)
    if flatten:
        return scaled_floats.flatten()
    else:
        return scaled_floats.reshape(img.size)


def list_to_img(img_as_list, img_size=None):
    """
    Can only handle 8-bit gray-scale
    :param img_as_list:
    :param img_size:
    :return: PIL image
    """
    assert type(img_as_list) == np.ndarray, type(img_as_list)
    if img_size is None:
        assert img_as_list.ndim == 2, ("Can't handle flattened arrays if not passed img_size", img_as_list.shape)
        img_size = img_as_list.shape
    scaled = img_as_list * (2**8 - 1)
    rounded = np.rint(scaled)
    ints = rounded.astype(dtype='uint8')
    img = Image.new('L', img_size)
    img.putdata(ints.flatten())
    return img


def shuffle_in_unison(a, b, random_permutation=True):
    """
    Shuffle two lists in unison
    :param a:
    :param b:
    :param random_permutation: if false, will use a static seed, effectively sorting the input arrays the same way each call
    :return:
    """
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    if not random_permutation:
        np.random.seed(123)  # only applies to one np.random call; calling np.random again will use a different seed
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def load(prep_funcs=(None,), prep_test=False, random_split=True):
    """
    Load images, convert to lists of scaled floats, split into training and test set, then expand training set by
    applying the pre-processing techniques in prep_funcs
    :param prep_funcs: List of pre-processing functions to apply to the images in the data set
    :param prep_test: Boolean whether or not the prep the test data set in addition to the training data set
    :param random_split: Boolean whether or not to split the data set into training and test set at a
                        random position (True) or at a fixed position(false)
    :return: namedtuple DataSet
    """
    assert len(prep_funcs) > 0
    x, y = load_images()

    split_index = int(len(x) * TEST_TO_TRAIN_RATIO)
    x_shuffled, y_shuffled = shuffle_in_unison(np.asarray(x, dtype=object), np.asarray(y), random_split)
    y_shuffled = convert_y(y_shuffled, N_CLASSES)
    # The y-values (labels) themselves are not prepped/changed in any way, but since there are more training examples
    # there has to be more labels.
    if prep_test:
        x_prepped = []
        y_prepped = []
        for prep_func in prep_funcs:
            if prep_func is None:
                x_prepped.extend(x_shuffled)
                y_prepped.extend(y_shuffled)
            else:
                x_prepped.extend(map(prep_func, x_shuffled))
                y_prepped.extend(y_shuffled)
        train_x = x_prepped[:split_index]
        train_y = y_prepped[:split_index]
        test_x = x_prepped[split_index:]
        test_y = y_prepped[split_index:]
    else:
        train_x = x_shuffled[:split_index]
        train_y = y_shuffled[:split_index]
        test_x = x_shuffled[split_index:]
        test_y = y_shuffled[split_index:]
        train_x_prepped = []
        train_y_prepped = []
        for prep_func in prep_funcs:
            if prep_func is None:
                train_x_prepped.extend(train_x)
                train_y_prepped.extend(train_y)
            else:
                train_y_prepped.extend(map(prep_func, train_x))
                train_y_prepped.extend(train_y)
        train_x = train_x_prepped
        train_y = train_y_prepped

    train_x = np.asarray([img_to_list(img, flatten=True) for img in train_x])
    test_x = np.asarray([img_to_list(img, flatten=True) for img in test_x])

    logging.debug('\tLoaded data set. Applied %s pre-processing function(s). Split data set with a ratio of %s into '
                  '%s training examples and %s test examples',
                  len(prep_funcs), TEST_TO_TRAIN_RATIO, len(train_x), len(test_x))
    #logging.debug("\tData set shapes: train_x %s, train_y %s, test_x %s, test_y %s",
    #              train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return DataSet(train_x, train_y, test_x, test_y)

