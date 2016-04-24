from collections import namedtuple
from PIL import Image, ImageFilter, ImageOps

DataSet = namedtuple("DataSet", 'train_x, train_y, test_x, test_y')


def edge_enhance(img):
    return img.filter(ImageFilter.EDGE_ENHANCE)


def edge_enhance_more(img):
    return img.filter(ImageFilter.EDGE_ENHANCE_MORE)


def invert(img):
    "Invert colors"
    return ImageOps.invert(img)


def mirror(img):
    "Mirror left to right"
    return ImageOps.mirror(img)


def i2c(intt):
    print(intt, type(intt))
    "convert integer to character"
    return chr(intt + ord('a'))
