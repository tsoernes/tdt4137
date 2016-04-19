import theano
from theano import tensor as T


def bp_sgd(cost, weights, l_rate, **kwargs):
    """
    Backpropagation function: Stochastic gradient descent
    """
    grads = T.grad(cost=cost, wrt=weights)
    updates = []
    for p, g in zip(weights, grads):
        updates.append([p, p - g * l_rate])
    return updates


def bp_rms_prop(cost, weights, l_rate, rho=0.9, epsilon=1e-6, **kwargs):
    """
    Backpropagation function: RMS-prop
    """
    grads = T.grad(cost, weights)
    updates = []
    for p, g in zip(weights, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - l_rate * g))
    return updates


def err_sum_squared(x, y):
    """
    Error function: Sum of the squared errors
    """
    return T.sum((x-y)**2)