from ann_utils import *
from ann import ANN
from theano import tensor as T
import theano
"""
ballpark learning rates for MNIST with different backprop funcs
BP_SDG_LR = 0.0015
BP_RMS_PROP_LR = 0.0006
"""

N_CLASSES = 26


def ann_preset_1():
    params = {
        'nodes_per_layer': [20**2, 25**2, N_CLASSES],
        'act_funcs': [T.nnet.sigmoid, T.nnet.softmax],
        'err_func': err_sum_squared,
        'backprop_func': bp_rms_prop,
        'backprop_params': {
            'rho': 0.9,
            'epsilon': 1e-6
        },
        "l_rate": 0.0008,
    }
    return params
