from net.net_utils import *
from net.convnet import ConvNet, FullyConnectedLayer, SoftMaxLayer, ConvPoolLayer
from theano import tensor as T
"""
ballpark learning rates for MNIST with different backprop funcs
BP_SDG_LR = 0.0015
BP_RMS_PROP_LR = 0.0006
"""

N_CLASSES = 26


def convnet_preset_1(batch_size=10):
    cp_layer1 = ConvPoolLayer(
                input_shape=(batch_size, 1, 20, 20),
                n_feature_maps=30,
                act_func=T.nnet.relu
    )
    cp_layer2 = ConvPoolLayer(
                input_shape=cp_layer1.output_shape,
                n_feature_maps=60,
                act_func=T.nnet.relu
    )
    params = {
        'layers': [
            cp_layer1,
            cp_layer2,
            FullyConnectedLayer(np.prod(cp_layer2.output_shape[1:]), 300, T.nnet.sigmoid),
            SoftMaxLayer(300, N_CLASSES)
        ],
        'err_func': T.nnet.categorical_crossentropy,
        #'err_func': err_sum_squared,
        'backprop_func': bp_rms_prop,
        'backprop_params': {
            'rho': 0.9,
            'epsilon': 1e-6
        },
        'l_rate': 0.0010,
        'batch_size': batch_size
    }
    return params


def convnet_preset_ffnet():
    """
    Same params as the ffnet, in order to compare performance between the two implementations
    """
    batch_size = 10
    params = {
        'layers': [
            FullyConnectedLayer(20*20, 25**2, T.nnet.sigmoid),
            SoftMaxLayer(25**2, N_CLASSES)
        ],
        'err_func': T.nnet.categorical_crossentropy,  # Categorical cross entropy = negative log likelihood
        'backprop_func': bp_rms_prop,
        'backprop_params': {
            'rho': 0.9,
            'epsilon': 1e-6
        },
        'l_rate': 0.0008,
        'batch_size': batch_size
    }
    return params

