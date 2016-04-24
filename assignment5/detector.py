import numpy as np
from PIL import Image
from PIL import ImageDraw
from load_prep import load_img, img_to_list, list_to_img
import logging
from utils import edge_enhance, edge_enhance_more, mirror, invert, i2c

MAX_WIDTH = MAX_HEIGHT = 200


def detect(ann, img, prep_funcs, window_size=20, stride=2, prob_threshold=0.8):
        # Create image patches
    assert len(prep_funcs) > 0
    logging.info("\tCreating image patches and testing their probabilities ...")
    patch_chars = []
    patch_probabilities = []
    box_positions = []
    for x in range(0, img.size[0]-window_size, stride):
        for y in range(0, img.size[1]-window_size, stride):
            image_patch_orig = img.crop((x, y, x+window_size, y+window_size))
            if window_size != 20:
                image_patch_orig = image_patch_orig.resize((20, 20), Image.ANTIALIAS)
            image_patches = []
            for prep_func in prep_funcs:
                if prep_func is None:
                    image_patches.append(image_patch_orig)
                else:
                    image_patches.append(prep_func(image_patch_orig))
            image_patches_l = [img_to_list(im, flatten=True) for im in image_patches]
            prediction = ann.predict(image_patches_l)
            for i in range(len(image_patches)):
                if prediction['char_probability'][i] > prob_threshold:
                    patch_chars.append(prediction['char_as_int'][i])
                    patch_probabilities.append(prediction['char_probability'][i])
                    box_positions.append([x, y])
                    break
        if x % (img.size[0] / 20) == 0:
            logging.info('\t\tProgress: %s%%', x/img.size[0]*100)

    logging.info("\tDetected %s characters with a probability of %s or higher:",
                 len(box_positions), prob_threshold)
    patch_probabilities = np.asarray(patch_probabilities).flatten()
    patch_chars = np.asarray(patch_chars).flatten()
    i_is = patch_probabilities.argsort()
    patch_probabilities = patch_probabilities[i_is]
    patch_chars = patch_chars[i_is]
    for i in range(len(patch_probabilities)):
        prob = patch_probabilities[i]
        logging.info('\t\t\tDetected character %c with probability %f',
                     chr(patch_chars[i] + ord('a')), prob)

    # Draw boxes around patches with highest probability
    draw_boxes(img, box_positions, window_size)


def resize(img_as_list, size=(20, 20)):
    """
    Take an image as a list of scaled floats, flattened or otherwise, and down/up sample to the given size
    :param img_as_list:
    :param size:
    :return:
    """
    img = list_to_img(img_as_list)
    img = img.resize(size, Image.ANTIALIAS)
    return img_to_list(img, flatten=False)


def draw_boxes(img, box_positions, window_size):
    """
    :param img:
    :param box_positions:
    :param window_size:
    :return:
    """
    draw = ImageDraw.Draw(img)
    for box_position in box_positions:
        draw.rectangle([box_position[0], box_position[1],
                        box_position[0]+window_size, box_position[1]+window_size],
                       fill=None, outline="red")
    del draw
    img.show()

