from ann_utils import *
from ann import ANN

"""
ballpark learning rates for MNIST with different backprop funcs
BP_SDG_LR = 0.0015
BP_RMS_PROP_LR = 0.0006
"""

N_CLASSES = 26


def ann_preset_1():
    #nodes_per_layer = [20 ** 2, 20 ** 2, N_CLASSES]
    nodes_per_layer = [20 ** 2, 40 ** 2, 20 ** 2, N_CLASSES*2, N_CLASSES]
    act_funcs = [T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.softmax]
    err_func = err_sum_squared
    backprop_func = bp_rms_prop
    backprop_params = {
        'rho': 0.9,
        'epsilon': 1e-6
    }
    l_rate = 0.0008
    net = ANN(nodes_per_layer, act_funcs, err_func, backprop_func, backprop_params, l_rate)
    return net
