import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageDraw
from load_prep import load_img,  img_to_list, list_to_img
import logging
import theano
from theano import tensor as T
from theano.tensor.signal.downsample import max_pool_2d
import numpy as np
import logging
from matplotlib import pyplot as plt
from net.net_utils import init_rand_weights, init_zero_weights

class ConvPoolLayer:
    conv_func = T.nnet.conv2d
    pool_func = max_pool_2d

    def __init__(self, image_shape, filter_shape, act_func,
                 local_receptive_field_size=(5,5), pool_size=(2,2),
                 init_weight_func=init_rand_weights, init_bias_weight_func=init_rand_weights):
        """
        Generate a convolutional and a subsequent pooling layer with one bias node for each channel in the pooling layer.
        :param image_shape: tuple(batch size, input channels, input rows, input columns) where
            input_channels = number of feature maps in upstream layer
            input rows, input columns = output size of upstream layer
        :param n_feature_maps: number of feature maps/filters in this layer
            filter rows, filter columns = size of local receptive field
        :param pool_size:
        :param act_func:
        :param init_weight_func:
        :param init_bias_weight_func:
        """
        self.image_shape = image_shape
        self.filter_shape = filter_shape
        self.n_feature_maps = filter_shape[0]
        self.act_func = act_func
        self.pool_size = pool_size
        self.weights = init_weight_func(self.filter_shape)
        self.bias_weights = init_bias_weight_func((self.n_feature_maps,))
        self.params = [self.weights, self.bias_weights]
        self.output_values = None

    def activate(self, input_values):
        """
        :param input_values: the output from the upstream layer (which is input to this layer)
        :return:
        """
        input_values = input_values.reshape(self.image_shape)
        conv = self.conv_func(
            input=input_values,
            filters=self.weights,
            filter_shape=self.filter_shape
        )
        pooled = self.pool_func(
            input=conv,
            ds=self.pool_size,
            ignore_border=True
        )
        self.output_values = self.act_func(pooled + self.bias_weights.dimshuffle('x', 0, 'x', 'x'))

    def output(self):
        assert self.output_values is not None, 'Asking for output before activating layer'
        return self.output_values


def test_conv_layer():
    batch_size = 10
    input_shape = (20, 20)
    image_shape = (batch_size, 1, 20, 20)
    filter_shape = (20, 1, 5, 5)
    n_feature_maps = 10
    convpool_layer = ConvPoolLayer(image_shape, filter_shape, T.nnet.relu)

    x = T.fmatrix('X')
    y = T.fmatrix('Y')

    convpool_layer.activate(x)


test_conv_layer()