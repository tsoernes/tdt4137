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
        prep_funcs = [edge_enhance_more, invert]
        self.data_set = load(prep_funcs)
        self.run_convnet()
        #self.run_ffnet()
        self.compare_samples_predictions(self.convnet)
        self.compare_samples_predictions(self.convnet)
        self.compare_samples_predictions(self.convnet)
        self.compare_samples_predictions(self.convnet)

    def run_ocr(self, ann):
        ocr_img_path = "./ocr-test4.jpg"
        # for window_size in range(58, 63):
        detect(ocr_img_path, ann, window_size=150, stride=10)

    def compare_run_times(self):
        print(timeit.timeit(self.run_convnet, number=1))
        print(timeit.timeit(self.run_ffnet, number=1))

    def run_convnet(self):
        ann_preset = convnet_preset_1()
        self.convnet = ConvNet(**ann_preset)
        self.convnet.train_and_test(self.data_set.train_x, self.data_set.train_y,
                                    self.data_set.test_x, self.data_set.test_y,
                                    epochs=20, plot=True)
        # 20 epochs sufficient

    def run_ffnet(self):
        ann_preset = ffnet_preset_1()
        self.ffnet = FFNet(**ann_preset)
        self.ffnet.train_and_test(self.data_set.train_x, self.data_set.train_y,
                                  self.data_set.test_x, self.data_set.test_y,
                                  epochs=30, batch_size=10, plot=True)

    def compare_samples_predictions(self, ann):
        """
        Take some samples from the test data set, print prediction and view corresponding images on screen
        :param n_samples:
        """
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
            img.save("./preds/pred"+pred+"cor"+cor+".jpg", "JPEG")

    @staticmethod
    def i2c(intt):
        "convert ingeter to character"
        return chr(intt + ord('a'))
runner = OCRRunner()
