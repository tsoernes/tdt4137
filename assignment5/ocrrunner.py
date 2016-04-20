from ann_preset import ann_preset_1
from load_prep import load, list_to_img
import logging
import numpy as np
from PIL import Image
from ann import ANN
from detector import detect


class OCRRunner:
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)
        ann_preset = ann_preset_1()
        self.data_set = load()
        self.ann = ANN(**ann_preset)
        self.ann.train_and_test(self.data_set.train_x, self.data_set.train_y,
                                self.data_set.test_x, self.data_set.test_y,
                                epochs=100, batch_size=100, plot=False)

        ocr_img_path = "./ocr-test4.png"
        #for window_size in range(58, 63):
        detect(ocr_img_path, self.ann, window_size=150, stride=10)

    def compare_samples_predictions(self, n_samples=5):
        """
        Take some samples from the test data set, print prediction and view corresponding images on screen
        :param n_samples:
        """
        samples_i = np.random.choice(len(self.data_set.test_x), size=n_samples)
        samples = self.data_set.test_x[samples_i]
        predictions = self.ann.predict(samples)['char_as_int']
        for sample, prediction in zip(samples, predictions):
            prediction = chr(prediction + ord('a'))
            logging.info("Predicted: %s", prediction)
            im = Image.new('L', (20, 20))
            im.putdata(list_to_img(sample))
            im.show()


runner = OCRRunner()
