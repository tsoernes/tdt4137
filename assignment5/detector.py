import numpy as np
from PIL import Image
from PIL import ImageDraw
from load_prep import load_img, img_to_list, list_to_img
import logging
from utils import edge_enhance, edge_enhance_more, mirror, invert

MAX_WIDTH = MAX_HEIGHT = 200
prep_funcs = [invert]

def detect(ann, img_path, resize_size=0, window_size=20, stride=2, prob_threshold=0.8):
    img = load_img(img_path)
    # Resize
    if resize_size > 0:
        img.thumbnail(resize_size, Image.ANTIALIAS)

    # Display image with a sample box in order to visually determine window_size
    logging.info("\tStarting OCR. Showing picture with example box of given window size")
    imc_c = img.copy()
    draw = ImageDraw.Draw(imc_c)
    draw.rectangle([0, 0, window_size, window_size], fill=None, outline="red")
    del draw
    imc_c.show()

    # Create image patches
    logging.info("\tCreating image patches and testing their probabilities ...")
    patch_chars = np.zeros(img.size)
    patch_probabilities = np.zeros(img.size)
    box_positions = []
    for x in range(0, img.size[0]-window_size, stride):
        for y in range(0, img.size[1]-window_size, stride):
            image_patch_orig = img.crop((x, y, x+window_size, y+window_size))
            if window_size != 20:
                image_patch_orig = image_patch_orig.resize((20, 20), Image.ANTIALIAS)
            image_patches = [image_patch_orig]
            for prep_func in prep_funcs:
                image_patches.append(prep_func(image_patch_orig))
            image_patches_l = [img_to_list(im, flatten=True) for im in image_patches]
            prediction = ann.predict(image_patches_l)
            for i in range(len(image_patches)):
                patch_chars[x][y] = prediction['char_as_int'][i]
                patch_probabilities[x][y] = prediction['char_probability'][i]
                if prediction['char_probability'][i] > prob_threshold:
                    logging.info('\t\t\tDetected character %c with probability %f at position x,y %i, %i',
                                 chr(prediction['char_as_int'][i] + ord('a')), prediction['char_probability'][i], x, y)
                    box_positions.append([x, y])
                    break
        if y % (img.size[0] / 10) == 0:
            logging.info('\t\tProgress: %s%%', y/img.size[0]*100)

    logging.info("\tDetected %s characters with a probability of %s or higher",
                 len(box_positions), prob_threshold)
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

