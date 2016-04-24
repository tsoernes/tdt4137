import logging

import numpy as np
from PIL import Image
from net.ffnet import FFNet
from net.convnet import ConvNet
from net.ffnet_preset import ffnet_preset_1
from net.convnet_preset import convnet_preset_1
from utils import edge_enhance, edge_enhance_more, mirror, invert, i2c
from detector import detect
from load_prep import load, list_to_img
import timeit
from load_prep import load_img
from PIL import ImageDraw


class OCRRunner:
    def __init__(self):
        self.ffnet = None
        self.convnet = None
        logging.basicConfig(level=logging.DEBUG)
        #prep_funcs = [edge_enhance, invert]
        self.prep_funcs = [None]
        self.data_set = load(self.prep_funcs)
        self.run_ocr()

    def run_ocr(self, img_path="./ocr-test6.png", resize_size=0, window_size=22, stride=2, prob_threshold=0.999):
        img = load_img(img_path)
        # Resize
        if resize_size > 0:
            img.thumbnail(resize_size, Image.ANTIALIAS)

        # Display image with a sample box in order to visually determine window_size
        logging.info("\tShowing picture with example box in top-left corner of given window size")
        imc_c = img.copy()
        draw = ImageDraw.Draw(imc_c)
        draw.rectangle([0, 0, window_size, window_size], fill=None, outline="red")
        del draw
        imc_c.show()

        self.run_convnet(batch_size=len(self.prep_funcs), plot=False, epochs=10)
        #self.run_ffnet(plot=False, epochs=20)
        ocr_prep_funcs = [edge_enhance]
        detect(self.convnet, img, ocr_prep_funcs, window_size, stride, prob_threshold)

    def compare_run_times(self):
        print(timeit.timeit(self.run_convnet, number=1))
        print(timeit.timeit(self.run_ffnet, number=1))

    def run_convnet(self, batch_size=10, plot=True, epochs=20):
        ann_preset = convnet_preset_1(batch_size)
        self.convnet = ConvNet(**ann_preset)
        self.convnet.train_and_test(self.data_set.train_x, self.data_set.train_y,
                                    self.data_set.test_x, self.data_set.test_y,
                                    epochs=epochs, plot=plot)
        # 20 epochs sufficient

    def run_ffnet(self, plot=True, epochs=30):
        ann_preset = ffnet_preset_1()
        self.ffnet = FFNet(**ann_preset)
        self.ffnet.train_and_test(self.data_set.train_x, self.data_set.train_y,
                                  self.data_set.test_x, self.data_set.test_y,
                                  epochs=epochs, batch_size=10, plot=plot)

    def compare_samples_predictions(self, show=True, save=False):
        """
        Take some samples from the test data set, print prediction and view corresponding images on screen
        :param show: display images
        :param save: save images
        """
        self.run_convnet()
        ann = self.convnet
        n_samples = 10
        samples_i = np.random.choice(len(self.data_set.test_x), size=n_samples)
        samples = self.data_set.test_x[samples_i]
        predictions = ann.predict(samples)
        for i in range(len(samples_i)):
            sample = self.data_set.test_x[samples_i[i]]
            pred = i2c(predictions['char_as_int'][i])  # prediction
            cor = i2c(np.argmax(self.data_set.test_y[samples_i[i]]))  # correct
            logging.info("Predicted: %s with probability %s. Correct classification: %s",
                         pred, predictions['char_probability'][i], cor)
            img = list_to_img(sample, img_size=(20, 20))
            if show:
                img.show()
            if save:
                img.save("./preds/"+pred+"-"+cor+" pred-correct.jpg", "JPEG")

if __name__ == "__main__":
    runner = OCRRunner()
