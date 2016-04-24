from theano import tensor as T

from net.net_utils import *

"""
ballpark learning rates for MNIST with different backprop funcs
BP_SDG_LR = 0.0015
BP_RMS_PROP_LR = 0.0006
"""

N_CLASSES = 26


def ffnet_preset_1():
    params = {
        'nodes_per_layer': [20**2, 25**2, N_CLASSES],
        'act_funcs': [T.nnet.sigmoid, T.nnet.softmax],
        'err_func': err_sum_squared,
        'backprop_func': bp_rms_prop,
        'backprop_params': {
            'rho': 0.9,
            'epsilon': 1e-6
        },
        "l_rate": 0.0015,
        "batch_size": 100
    }
    return params
