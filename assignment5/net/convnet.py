import theano
from theano import tensor as T
from theano.tensor.signal.downsample import max_pool_2d
import numpy as np
import logging
from matplotlib import pyplot as plt
from net.net_utils import init_rand_weights, init_zero_weights


class ConvNet:
    def __init__(self, layers, err_func, backprop_func, backprop_params,
                 l_rate=.001, batch_size=10):
        """
        err_func - cost/error function.
        backprop_func - backpropagation function.
        l_rate - Learning rate
        """
        self.batch_size = batch_size
        logging.info('\tConstructing ANN with nodes per layer: learning rate: %s ', l_rate)
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
        prediction = T.argmax(output_layer, axis=1)
        prediction_value = T.max(output_layer, axis=1)
        updates = backprop_func(cost, params, l_rate, **backprop_params)

        # logging.info('\tConstructing functions ...')
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
        assert len(train_x) == len(train_y), (len(train_x), len(train_y))
        assert len(test_x) == len(test_y), (len(test_x), len(test_y))
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
            tr_result = np.zeros(shape=(len(train_x)))
            te_result = np.zeros(shape=(len(test_x)))
            for k in range(n_training_batches):
                tr_result[k*self.batch_size:(k+1)*self.batch_size] = \
                    self.predictor(train_x[k*self.batch_size:(k+1)*self.batch_size])['char_as_int']
            for l in range(n_testing_batches):
                batch = test_x[l*self.batch_size:(l+1)*self.batch_size]
                te_result[l*self.batch_size:(l+1)*self.batch_size] = self.predictor(batch)['char_as_int']
                # logging.debug('\t\t\t\t L:%s:%s / %s, batch size %s', l, l+self.batch_size, len(test_x), len(batch))
            tr_success_rate = np.mean(np.argmax(train_y, axis=1) == tr_result)
            te_success_rate = np.mean(np.argmax(test_y, axis=1) == te_result)
            train_success_rates.append(tr_success_rate)
            test_success_rates.append(te_success_rate)

            if i % (epochs / 5) == 0:
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
    def __init__(self, n_in, n_out, act_func,
                 init_weight_func=init_rand_weights, init_bias_weight_func=init_rand_weights):
        """
        Generate a fully connected layer with 1 bias node simulated upstream
        :param act_func: the activation function of the layer
        """
        self.n_in = n_in
        self.n_out = n_out
        self.act_func = act_func
        self.weights = init_weight_func((n_in, n_out))
        self.bias_weights = init_bias_weight_func((n_out,))
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
        super(SoftMaxLayer, self).__init__(n_in, n_out, T.nnet.softmax, init_zero_weights, init_zero_weights)


class ConvPoolLayer:
    conv_func = T.nnet.conv2d
    pool_func = max_pool_2d

    def __init__(self, image_shape, n_feature_maps, act_func,
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
        self.filter_shape = (n_feature_maps, image_shape[1]) + local_receptive_field_size
        self.act_func = act_func
        self.pool_size = pool_size
        self.weights = init_weight_func(self.filter_shape)
        self.bias_weights = init_bias_weight_func((n_feature_maps,))
        self.params = [self.weights, self.bias_weights]
        self.output_values = None

    def activate(self, input_values, *args):
        """
        :param input_values: the output from the upstream layer (which is input to this layer)
        :return:
        """
        #input_values = input_values.reshape(self.image_shape)
        conv = self.conv_func(
            input=input_values,
            image_shape=self.image_shape,
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

