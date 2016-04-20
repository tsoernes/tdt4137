import numpy as np
from PIL import Image
from PIL import ImageDraw
from load_prep import load_img, img_to_list, list_to_img
import logging


PATCH_PROB_THRESHOLD = 0.99
MAX_WIDTH = MAX_HEIGHT = 200


def detect(img_path, ann, window_size=20, stride=2):
    img = load_img(img_path)
    # Resize
    # img_as_list = thumbnail(img_as_list, (MAX_WIDTH, MAX_HEIGHT))

    # Create image patches
    logging.info("\tCreating image patches and testing their probabilities ...")
    patch_chars = np.zeros(img.size)
    patch_probabilities = np.zeros(img.size)
    box_positions = []
    for x in range(0, img.size[0]-window_size, stride):
        for y in range(0, img.size[1]-window_size, stride):
            image_patch = img.crop((x, y, x+window_size, y+window_size))
            if window_size != 20:
                image_patch = image_patch.resize((20, 20), Image.ANTIALIAS)
            prediction = ann.predict(np.asarray([img_to_list(image_patch, flatten=True)]))
            patch_chars[x][y] = prediction['char_as_int'][0]
            patch_probabilities[x][y] = prediction['char_probability'][0]
            if prediction['char_probability'][0] > PATCH_PROB_THRESHOLD:
                logging.info('\t\t\tDetected character %c with probability %f at position x,y %i, %i',
                             chr(prediction['char_as_int'][0] + ord('a')), prediction['char_probability'][0], x, y)
                box_positions.append([x, y])
        if y % (img.size[0] / 10) == 0:
            logging.info('\t\tProgress: %s%%', y/img.size[0]*100)

    logging.info("\tDetected %s characters with a probability of %s or higher",
                 len(box_positions), PATCH_PROB_THRESHOLD)
    # Draw boxes around patches with highest probability
    draw_boxes(img, box_positions, window_size)


def detect2(img_path, ann, window_size=20, stride=2):
    img_as_list = img_path_to_list(img_path, flatten=False)
    img_as_list = thumbnail(img_as_list, (MAX_WIDTH, MAX_HEIGHT))

    # Create image patches
    logging.info("\tCreating image patches and testing their probabilities ...")
    patch_chars = np.zeros(img_as_list.shape)
    patch_probabilities = np.zeros(img_as_list.shape)
    box_positions = []
    for y in range(0, len(img_as_list)-window_size, stride):
        for x in range(0, len(img_as_list[y])-window_size, stride):
            image_patch = img_as_list[y:y+window_size, x:x+window_size]
            if window_size != 20:
                image_patch = image_patch.resize((20, 20), Image.ANTIALIAS)
            prediction = ann.predict(np.asarray([image_patch.flatten()]))
            patch_chars[y][x] = prediction['char_as_int'][0]
            patch_probabilities[y][x] = prediction['char_probability'][0]
            if prediction['char_probability'][0] > PATCH_PROB_THRESHOLD:
                logging.info('\t\t\tDetected character %c with probability %f at position x,y %i, %i',
                             chr(prediction['char_as_int'][0] + ord('a')), prediction['char_probability'][0], x, y)
                box_positions.append([y, x])
        if y % (len(img_as_list) / 10) == 0:
            logging.info('\t\tProgress: %s%%', y/len(img_as_list)*100)

    logging.info("\tDetected %s characters with a probability of %s or higher",
                 len(box_positions), PATCH_PROB_THRESHOLD)
    # Draw boxes around patches with highest probability
    draw_boxes(list_to_img(img_as_list), box_positions, window_size)


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


def thumbnail(img_as_list, size):
    img = list_to_img(img_as_list)
    img.thumbnail(size, Image.ANTIALIAS)
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

