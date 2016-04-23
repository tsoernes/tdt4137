import logging

import numpy as np
from PIL import Image
from net.ffnet import FFNet
from net.convnet import ConvNet
from net.ffnet_preset import ffnet_preset_1
from net.convnet_preset import convnet_preset_1, convnet_preset_2
from utils import edge_enhance, edge_enhance_more, mirror, invert
from detector import detect
from load_prep import load, list_to_img
import timeit


class OCRRunner:
    def __init__(self):
        self.ffnet = None
        self.convnet = None
        logging.basicConfig(level=logging.DEBUG)
        #prep_funcs = [edge_enhance, invert]
        prep_funcs = [invert]
        self.data_set = load(prep_funcs)

    def run_ocr(self, img_path="./ocr-test4.jpg", resize_size=0, window_size=20, stride=2, prob_threshold=0.9):
        self.run_convnet(batch_size=2, plot=False)
        # for window_size in range(58, 63):
        detect(self.convnet, img_path, resize_size, window_size, stride, prob_threshold)

    def compare_run_times(self):
        print(timeit.timeit(self.run_convnet, number=1))
        print(timeit.timeit(self.run_ffnet, number=1))

    def run_convnet(self, batch_size=10, plot=True):
        ann_preset = convnet_preset_1(batch_size)
        self.convnet = ConvNet(**ann_preset)
        self.convnet.train_and_test(self.data_set.train_x, self.data_set.train_y,
                                    self.data_set.test_x, self.data_set.test_y,
                                    epochs=20, plot=plot)
        # 20 epochs sufficient

    def run_ffnet(self):
        ann_preset = ffnet_preset_1()
        self.ffnet = FFNet(**ann_preset)
        self.ffnet.train_and_test(self.data_set.train_x, self.data_set.train_y,
                                  self.data_set.test_x, self.data_set.test_y,
                                  epochs=30, batch_size=10, plot=True)

    def compare_samples_predictions(self, show=True, save=False):
        """
        Take some samples from the test data set, print prediction and view corresponding images on screen
        :param n_samples:
        """
        self.run_convnet()
        ann = self.convnet
        n_samples = 10
        samples_i = np.random.choice(len(self.data_set.test_x), size=n_samples)
        samples = self.data_set.test_x[samples_i]
        predictions = ann.predict(samples)
        for i in range(len(samples_i)):
            sample = self.data_set.test_x[samples_i[i]]
            pred = self.i2c(predictions['char_as_int'][i]) # prediction
            cor = self.i2c(np.argmax(self.data_set.test_y[samples_i[i]])) # correct
            logging.info("Predicted: %s with probability %s. Correct classification: %s",
                         pred, predictions['char_probability'][i], cor)
            img = list_to_img(sample, img_size=(20, 20))
            if show:
                img.show()
            if save:
                img.save("./preds/"+pred+"-"+cor+" pred-correct.jpg", "JPEG")

    @staticmethod
    def i2c(intt):
        "convert ingeter to character"
        return chr(intt + ord('a'))

runner = OCRRunner()
