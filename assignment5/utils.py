from collections import namedtuple
from PIL import Image, ImageFilter

DataSet = namedtuple("DataSet", 'train_x, train_y, test_x, test_y')


def edge_enhance(img):
    return img.filter(ImageFilter.EDGE_ENHANCE)


def edge_enhance_more(img):
    return img.filter(ImageFilter.EDGE_ENHANCE_MORE)
