import logging

import theano
from theano import tensor as T
import numpy as np

from net.ann import ANN
from net.net_utils import init_rand_weights


class FFNet(ANN):
    def __init__(self, nodes_per_layer, act_funcs, err_func, backprop_func, backprop_params,
                 l_rate=.001, batch_size=100):
        """
        layer_shape - number of nodes per layer, including input and output layers
        act_funcs - list activation functions between the layers
        err_func - cost/error function
        backprop_func - backpropagation function
        l_rate - Learning rate
        """
        assert len(nodes_per_layer)-1 == len(act_funcs), \
            ("Invalid number of activation functions compared to the number of hidden layers",
             len(nodes_per_layer), len(act_funcs))
        super(FFNet, self).__init__('FFNet', l_rate, batch_size)
        logging.info('\tConstructing FFNet with nodes per layer: %s, learning rate: %s ', nodes_per_layer, l_rate)

        input_data = T.fmatrix('X')
        input_labels = T.bmatrix('Y')
        layers = [input_data]

        # Generate initial random weights between each layer
        weights = []
        for i in range(len(nodes_per_layer)-1):
            weights.append(init_rand_weights((nodes_per_layer[i], nodes_per_layer[i+1])))
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
        updates = backprop_func(cost, weights, self.l_rate, **backprop_params)

        prediction = T.argmax(output_layer, axis=1)
        prediction_value = T.max(output_layer, axis=1)

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

    def _train(self, input_data, input_labels):
        return self.trainer(input_data, input_labels)

    def predict(self, input_data):
        return self.predictor(input_data)

    @staticmethod
    def model(upstream_layer, weights, act_func):
        """
        Generate a layer given the activation function of the layer, the upstream layer and the weights
        going from the upstream layer to the new one
        """
        layer = act_func(T.dot(upstream_layer, weights))
        return layer



