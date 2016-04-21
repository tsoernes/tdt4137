import theano
from theano import tensor as T
import numpy as np


def bp_sgd(cost, params, l_rate, **kwargs):
    """
    Backpropagation function: Stochastic gradient descent
    """
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g*l_rate])
    return updates


def bp_rms_prop(cost, weights, l_rate, rho=0.9, epsilon=1e-6, **kwargs):
    """
    Backpropagation function: RMS-prop
    """
    grads = T.grad(cost, weights)
    updates = []
    for p, g in zip(weights, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g**2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - l_rate * g))
    return updates


def err_sum_squared(output_layer, input_labels, *args):
    """
    Error function: Sum of the squared errors
    """
    return T.sum((output_layer - input_labels) ** 2)


def err_neg_log_likelihood(input_labels, layers, n_batches, l2_norm_squared, lmbda=0.0):

    cost = -T.mean([T.arange(input_labels.shape[0]), input_labels]) + 0.5 * lmbda * l2_norm_squared / n_batches
    return cost


def init_rand_weights(shape, name=""):
    """
    Generate random weights (from the standard normal distribution, scaled) in an x-by-y matrix.
    Should be used by most layers.
    """
    return theano.shared(
        np.asarray(np.random.randn(*shape) * 0.01, dtype=theano.config.floatX), name=name)


def init_zero_weights(shape, name=""):
    """
    Generate weighs with value 0. Should be used for Softmax layers.
    """
    return theano.shared(np.zeros(shape, dtype=theano.config.floatX), name=name)