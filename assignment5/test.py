import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageDraw
from load_prep import load_img,  img_to_list, list_to_img
import logging


PATCH_PROB_THRESHOLD = 0.7
MAX_WIDTH = MAX_HEIGHT = 200


ocr_img_path = "./ocr-test.jpg"
img_path = "./chars74k-lite/chars74k-lite/a/a_0.jpg"
img1 = load_img(img_path)
img2 = img1.filter(ImageFilter.EDGE_ENHANCE)
img3 = img1.filter(ImageFilter.EDGE_ENHANCE_MORE)
img1.save('./none.jpg', "JPEG")
img2.save('./edge.jpg', "JPEG")
img3.save('./edgemore.jpg', "JPEG")
