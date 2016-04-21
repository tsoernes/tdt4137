from net.net_utils import *
from net.convnet import ConvNet, FullyConnectedLayer, SoftMaxLayer, ConvPoolLayer
"""
ballpark learning rates for MNIST with different backprop funcs
BP_SDG_LR = 0.0015
BP_RMS_PROP_LR = 0.0006
"""

N_CLASSES = 26


def convet_preset_1():
    batch_size = 10
    params = {
        'layers': [
            ConvPoolLayer(
                image_shape=(batch_size, 1, 20, 20),
                n_feature_maps=40,
                act_func=T.nnet.relu
            ),
            FullyConnectedLayer(20**2, 100, T.nnet.sigmoid),
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
