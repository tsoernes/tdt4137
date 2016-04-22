from net.net_utils import *
from net.convnet import ConvNet, FullyConnectedLayer, SoftMaxLayer, ConvPoolLayer
"""
ballpark learning rates for MNIST with different backprop funcs
BP_SDG_LR = 0.0015
BP_RMS_PROP_LR = 0.0006
"""

N_CLASSES = 26


def convnet_preset_1():
    batch_size = 10
    cp_layer1 = ConvPoolLayer(
                input_shape=(batch_size, 1, 20, 20),
                n_feature_maps=20,
                act_func=T.nnet.relu
    )
    cp_layer2 = ConvPoolLayer(
                input_shape=cp_layer1.get_output_shape(),
                n_feature_maps=40,
                act_func=T.nnet.relu
    )
    params = {
        'layers': [
            cp_layer1,
            cp_layer2,
            FullyConnectedLayer(cp_layer2.get_output_shape(), 100, T.nnet.sigmoid),
            SoftMaxLayer(100, N_CLASSES)
        ],
        'err_func': err_sum_squared,
        'backprop_func': bp_rms_prop,
        'backprop_params': {
            'rho': 0.9,
            'epsilon': 1e-6
        },
        'l_rate': 0.0005,
        'batch_size': batch_size
    }
    return params


def convnet_preset_2():
    # Todo: check and infer layer input/output sizes and params
    batch_size = 10
    params = {
        'layers': [
            ConvPoolLayer(
                input_shape=(batch_size, 1, 20, 20),
                n_feature_maps=15,
                act_func=T.nnet.relu
            ),
            FullyConnectedLayer(15*8*8, 100, T.nnet.sigmoid),
            SoftMaxLayer(100, N_CLASSES)
        ],
        'err_func': err_sum_squared,
        'backprop_func': bp_rms_prop,
        'backprop_params': {
            'rho': 0.9,
            'epsilon': 1e-6
        },
        'l_rate': 0.0005,
        'batch_size': batch_size
    }
    return params


def convnet_preset_ffnet():
    batch_size = 10
    params = {
        'layers': [
            FullyConnectedLayer(20*20, 25**2, T.nnet.sigmoid),
            SoftMaxLayer(25**2, N_CLASSES)
        ],
        'err_func': err_sum_squared,
        'backprop_func': bp_rms_prop,
        'backprop_params': {
            'rho': 0.9,
            'epsilon': 1e-6
        },
        'l_rate': 0.0008,
        'batch_size': batch_size
    }
    return params

