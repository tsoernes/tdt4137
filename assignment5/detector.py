import numpy as np
from ann import ann
from PIL import Image
from PIL import ImageDraw
from load_prep import list_to_img
import logging


class Detector:
    PATCH_PROB_THRESHOLD = 0.8

    def detect(self, img_path, ann, window_size=20):
        img_list = self.load_image(img_path)
        # Create image patches
        image_patches = []
        for y in range(len(img_list)-window_size):
            for x in range(len(img_list[y])-window_size):
                image_patch = []  # flattened
                for y_patch in range(y, y+window_size):
                    for x_patch in range(x, x+window_size):
                        image_patch.append(img_list[y_patch][x_patch])
                image_patches.append(image_patch)

        # Resize patches to fit ANN input, if necessary
        if window_size != 20:
            image_patches = map(self.resize, image_patches)

        # Get max probability for each patch using ANN
        prediction = ann.predict(image_patches)
        patch_chars = prediction[0].reshape(img_list.size)
        patch_probs = prediction[1].reshape(img_list.size)
        print(patch_chars, patch_probs)

        # Draw boxes around patches with highest probability
        box_positions = []
        for y in range(len(patch_probs)):
            for x in range(len(patch_probs[y])):
                if patch_probs[y][x] > self.PATCH_PROB_THRESHOLD:
                    box_positions.append([y, x])
        print(box_positions)
        self.draw_boxes(list_to_img(img_list), box_positions, window_size)

    @staticmethod
    def load_image(img_path, nbits=8):
        normalized_floats = np.asarray(Image.open(img_path)).astype(float) / (2**nbits-1)
        return normalized_floats

    @staticmethod
    def resize(img_list, size=(20, 20)):
        """
        Take an image as a list of scaled floats, flattened or otherwise, and downsample to the given size
        :param img_list:
        :param size:
        :return:
        """
        img = list_to_img(img_list)
        img = img.resize(size,Image.ANTIALIAS)
        return img

    @staticmethod
    def draw_boxes(img, box_positions, window_size):
        """
        :param img:
        :param box_positions:
        :param window_size:
        :return:
        """
        draw = ImageDraw.Draw(img)
        for box_position in box_positions:
            draw.rectangle([box_position[1], box_position[0], box_position[1]+window_size, box_position[0]+window_size],
                           fill=2, outline="red")
        img.show()

