import theano
from theano import tensor as T
from theano.tensor.signal.downsample import max_pool_2d  # deprecated
from theano.tensor.signal.pool import pool_2d
import numpy as np
import logging
from matplotlib import pyplot as plt
from net.net_utils import init_rand_weights, init_zero_weights, err_neg_log_likelihood


class ConvNet:
    def __init__(self, layers, err_func, backprop_func, backprop_params,
                 l_rate, batch_size=10):
        """
        :param layers:
        :param err_func: cost/error function
        :param backprop_func: backpropagation function
        :param backprop_params: parameters to pass to backprop function
        :param l_rate: learning rate
        :param batch_size: (mini-) batch size. In comparison to regular nets
        :return:
        """
        self.batch_size = batch_size
        logging.info('\tConstructing ANN with %s layers. Learning rate: %s ', len(layers), l_rate)

        params = []  # Regular weights and bias weights; e.g. everything to be adjusted during training
        for layer in layers:
            for param in layer.params:
                params.append(param)
        logging.info('\tNumber of parameters to train: %s',
                     sum(param.get_value(borrow=True, return_internal_type=True).size for param in params))

        input_data = T.fmatrix('X')
        input_labels = T.fmatrix('Y')

        layers[0].activate(input_data, batch_size)
        for i in range(1, len(layers)):
            prev_layer = layers[i-1]
            current_layer = layers[i]
            current_layer.activate(prev_layer.output(), batch_size)

        output_layer = layers[-1].output_values
        cost = err_func(output_layer, input_labels)
        updates = backprop_func(cost, params, l_rate, backprop_params)

        prediction = T.argmax(output_layer, axis=1)
        prediction_value = T.max(output_layer, axis=1)

        logging.debug('\tConstructing functions ...')
        self.trainer = theano.function(
            inputs=[input_data, input_labels],
            outputs=cost,
            updates=updates,
            name='Trainer',
            allow_input_downcast=True  # Allows float64 to be casted as float32, which is necessary in order to use GPU
        )
        self.predictor = theano.function(
            inputs=[input_data],
            outputs={'char_as_int': prediction,
                     'char_probability': prediction_value,
                     'output_layer': output_layer},
            name='Predictor',
            allow_input_downcast=True
        )

    def train_and_test(self, train_x, train_y, test_x, test_y, epochs=200, plot=True):
        assert len(train_x) == len(train_y) and len(test_x) == len(test_y), \
            ("Training", len(train_x), len(train_y), "or testing", len(test_x), len(test_y),
             "data sets does not have the same amount of data as classifications")
        logging.info('\tTraining and testing for %s epochs ...', epochs)

        """
        Both training and testing is done in batches for increased speed and to avoid running out of memory.
        If len(train_x) % self.batch size != 0 then some training examples will not be trained on. The same applies to
        testing.
        """
        n_training_batches = len(train_x) // self.batch_size
        n_testing_batches = len(test_x) // self.batch_size
        train_success_rates = []
        test_success_rates = []
        for i in range(epochs):
            for j in range(n_training_batches):
                x_cases = train_x[j*self.batch_size:(j+1)*self.batch_size]
                y_cases = train_y[j*self.batch_size:(j+1)*self.batch_size]
                self.trainer(x_cases, y_cases)

            # Get success rate on training and test data set
            tr_result = np.zeros(shape=(n_training_batches*self.batch_size))
            te_result = np.zeros(shape=(n_testing_batches*self.batch_size))
            for k in range(n_training_batches):
                batch = train_x[k*self.batch_size:(k+1)*self.batch_size]
                tr_result[k*self.batch_size:(k+1)*self.batch_size] = self.predictor(batch)['char_as_int']
            for l in range(n_testing_batches):
                batch = test_x[l*self.batch_size:(l+1)*self.batch_size]
                te_result[l*self.batch_size:(l+1)*self.batch_size] = self.predictor(batch)['char_as_int']
                # logging.debug('\t\t\t\t L:%s:%s / %s, batch size %s', l, l+self.batch_size, len(test_x), len(batch))
            # todo: verify that the length of each comparison result set is equal,
            # and that the sets are equally 'full' (no missing values)
            tr_success_rate = np.mean(np.argmax(train_y[:n_training_batches*self.batch_size], axis=1) == tr_result)
            te_success_rate = np.mean(np.argmax(test_y[:n_testing_batches*self.batch_size], axis=1) == te_result)
            train_success_rates.append(tr_success_rate)
            test_success_rates.append(te_success_rate)

            if i % (epochs / 20) == 0:
                logging.info('\t\tProgress: %s%% | Epoch: %s | Success rate (training, test): %s, %s',
                             (i / epochs)*100, i,
                             "{:.4f}".format(max(train_success_rates)), "{:.4f}".format(max(test_success_rates)))

        logging.info('\tMax success rate (training | test): %s | %s',
                     "{:.4f}".format(max(train_success_rates)), "{:.4f}".format(max(test_success_rates)))
        if plot:
            plt.title('Convolutional Pooled Net')
            plt.plot(train_success_rates)
            plt.plot(test_success_rates)
            plt.legend(['Train', 'Test'], loc="best")
            plt.grid(True)
            plt.yticks(np.arange(0, 1, 0.05))
            plt.show()

    def predict(self, input_x):
        return self.predictor(input_x)


class FullyConnectedLayer:
    def __init__(self, n_in, n_out, act_func):
        """
        Generate a fully connected layer with 1 bias node simulated upstream
        :param act_func: the activation function of the layer
        """
        self.n_in = n_in
        self.n_out = n_out
        self.act_func = act_func
        self.weights = init_rand_weights((n_in, n_out), "w")
        self.bias_weights = init_rand_weights((n_out,), "b")
        self.params = [self.weights, self.bias_weights]
        self.output_values = None

    def activate(self, input_values, batch_size):
        """
        :param input_values: the output from the upstream layer (which is input to this layer)
        :param batch_size:
        :return:
        """
        input_values = input_values.reshape((batch_size, self.n_in))
        self.output_values = self.act_func(T.dot(input_values, self.weights) + self.bias_weights)

    def output(self):
        assert self.output_values is not None, 'Asking for output before activating layer'
        return self.output_values


class SoftMaxLayer(FullyConnectedLayer):
    def __init__(self, n_in, n_out):
        super(SoftMaxLayer, self).__init__(n_in, n_out, T.nnet.softmax)
        # todo find out how to initialize softmaxlayer weights as zero, and if it gives better results/faster training


class ConvPoolLayer:
    conv_func = staticmethod(T.nnet.conv2d)
    pool_func = staticmethod(pool_2d)

    def __init__(self, input_shape, n_feature_maps, act_func,
                 local_receptive_field_size=(5, 5), pool_size=(2, 2), pool_mode='max'):
        """
        Generate a convolutional and a subsequent pooling layer with one bias node for each channel in the pooling layer.
        :param input_shape: tuple(batch size, input channels, input rows, input columns) where
            input_channels = number of feature maps in upstream layer
            input rows, input columns = output size of upstream layer
        :param n_feature_maps: number of feature maps/filters in this layer
        :param local_receptive_field_size: (filter rows, filter columns) = size of local receptive field
        :param pool_size: (rows, columns)
        :param act_func: activation function to be applied to the output from the pooling layer
        :param init_weight_func:
        :param init_bias_weight_func:
        """
        self.input_shape = input_shape
        self.n_feature_maps = n_feature_maps
        self.filter_shape = (n_feature_maps, input_shape[1]) + local_receptive_field_size
        self.local_receptive_field_size = local_receptive_field_size
        self.act_func = act_func
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.weights = init_rand_weights(self.filter_shape, "conv2poolWeights")
        self.bias_weights = init_rand_weights((n_feature_maps,), "conv2poolBiasWeights")
        self.params = [self.weights, self.bias_weights]
        self.output_values = None

    def activate(self, input_values, *args):
        """
        :param input_values: the output from the upstream layer (which is input to this layer)
        :return:
        """
        input_values = input_values.reshape(self.input_shape)
        conv = self.conv_func(
            input=input_values,
            input_shape=self.input_shape,
            filters=self.weights,
            filter_shape=self.filter_shape
        )
        pooled = self.pool_func(
            input=conv,
            ds=self.pool_size,
            ignore_border=True,  # If the pool size does not evenly divide the input,
                                 # then ignoring the border will pool from padded zeros.
                                 # This is usually the desired behaviour when max pooling.
            # st=(1,1) # Stride size. Defaults to pool size, e.g. non-overlapping pooling regions
            mode=self.pool_mode  # ‘max’, ‘sum’, ‘average_inc_pad’ or ‘average_exc_pad’
        )
        self.output_values = self.act_func(pooled + self.bias_weights.dimshuffle('x', 0, 'x', 'x'))

    def output(self):
        assert self.output_values is not None, 'Asking for output before activating layer'
        return self.output_values

    def get_output_shape(self):
        batch_size = self.input_shape[0]
        if self.local_receptive_field_size[0] != self.local_receptive_field_size[1] \
                or self.pool_size[0] != self.pool_size[1]:
            raise NotImplementedError("I don't know how to calculate output shape when the local receptive field,",
                                      self.local_receptive_field_size, ", or the pool,",
                                      self.pool_size, ", is non-square")
        after_conv = self.input_shape[2] - self.local_receptive_field_size[0]
        after_pool = np.ceil(after_conv/2.0)
        shape = (batch_size, self.n_feature_maps, after_pool, after_pool)
        return shape

