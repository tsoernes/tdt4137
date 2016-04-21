import theano
from theano import tensor as T
import numpy as np
import logging
from matplotlib import pyplot as plt


class FFNet:
    def __init__(self, nodes_per_layer, act_funcs, err_func, backprop_func, backprop_params, l_rate=.001):
        """
        layer_shape - number of nodes per layer, including input and output layers
        act_funcs - list activation functions between the layers. Examples: T.nnet.sigmoid, .softmax, .relu, .categorial_crossentropy
        err_func - cost/error function. Example: lambda x,y: T.mean(T.nnet.categorical_crossentropy(x, y)).
        backprop_func - backpropagation function. Example:
        l_rate - Learning rate
        """
        assert len(nodes_per_layer)-1 == len(act_funcs), \
            ("Invalid number of activation functions compared to the number of hidden layers",
             len(nodes_per_layer), len(act_funcs))

        logging.info('\tConstructing ANN with nodes per layer: %s, learning rate: %s ', nodes_per_layer, l_rate)

        input_data = T.fmatrix('X')
        input_labels = T.fmatrix('Y')
        layers = [input_data]

        # Generate initial random weights between each layer
        weights = []
        for i in range(len(nodes_per_layer)-1):
            weights.append(self.init_rand_weights(nodes_per_layer[i], nodes_per_layer[i+1]))
            weights[i].name = 'w' + str(i)
        # logging.debug('\tWeight layers: %s', len(weights))
        #logging.info('\tNumber of parameters to train: %s',
        #             sum(param.get_value(borrow=True, return_internal_type=True).size for param in weights))
        # Construct the layers with the given activation functions weights between them
        # logging.info('\tConstructing layers ...')
        for i in range(len(weights)):
            layers.append(self.model(layers[i], weights[i], act_funcs[i]))

        for i in range(1, len(layers)):
            layers[i].name = 'l' + str(i)

        output_layer = layers[-1]
        cost = err_func(output_layer, input_labels)
        prediction = T.argmax(output_layer, axis=1)
        prediction_value = T.max(output_layer, axis=1)
        updates = backprop_func(cost, weights, l_rate, **backprop_params)

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

    def train_and_test(self, train_x, train_y, test_x, test_y, epochs=200, batch_size=150, plot=True):
        assert len(train_x) == len(train_y), ("", len(train_x), len(train_y))
        logging.info('\tTraining and testing for %s epochs ...', epochs)
        train_success_rates = []
        test_success_rates = []
        for i in range(epochs):
            # Run in batches for increased speed and to avoid running out of memory
            for j in range(0, len(train_x), batch_size):
                x_cases = train_x[j:j+batch_size]
                y_cases = train_y[j:j+batch_size]
                self.trainer(x_cases, y_cases)

            tr_result = np.zeros(shape=(len(train_x)))
            for k in range(0, len(train_x), batch_size):
                tr_result[k:k+batch_size] = self.predictor(train_x[k:k+batch_size])['char_as_int']
            # Get success rate on training and test data set
            tr_success_rate = np.mean(np.argmax(train_y, axis=1) == tr_result)
            te_success_rate = np.mean(np.argmax(test_y, axis=1) == self.predictor(test_x)['char_as_int'])
            train_success_rates.append(tr_success_rate)
            test_success_rates.append(te_success_rate)

            if i % (epochs / 5) == 0:
                logging.info('\t\tProgress: %s%% | Epoch: %s | Success rate (training, test): %s, %s',
                             (i / epochs)*100, i,
                             "{:.4f}".format(max(train_success_rates)), "{:.4f}".format(max(test_success_rates)))

        logging.info('\tMax success rate (training | test): %s | %s',
                     "{:.4f}".format(max(train_success_rates)), "{:.4f}".format(max(test_success_rates)))
        if plot:
            plt.title('Fully Connected Feed Forward Net')
            plt.plot(train_success_rates)
            plt.plot(test_success_rates)
            plt.legend(['Train', 'Test'], loc="best")
            plt.grid(True)
            plt.yticks(np.arange(0, 1, 0.05))
            plt.show()

    def predict(self, input_x):
        return self.predictor(input_x)

    @staticmethod
    def init_rand_weights(x, y):
        """
        Generate random weights (from the standard normal distribution, scaled) in an x-by-y matrix.
        """
        return theano.shared(
            np.asarray(np.random.randn(x, y) * 0.01, dtype=theano.config.floatX))

    @staticmethod
    def model(upstream_layer, weights, act_func):
        """
        Generate a layer given the activation function of the layer, the upstream layer and the weights
        going from the upstream layer to the new one
        """
        layer = act_func(T.dot(upstream_layer, weights))
        return layer

    @staticmethod
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]


