import numpy as np


def inverse_softplus(x):
    return np.log(np.exp(x) - 1.)


def logit(x):
    """
    This is equivalent to the inverse of nn.functional.sigmoid()
    :param x:
    :return:
    """
    return np.log(x / (1 - x))


def sigmoid_inv(x):
    return logit(x)


def softplus(x, limit=30):
    if x > limit:
        return x
    else:
        return np.log(1.0 + np.exp(x))


def softplus2(x):
    return softplus(x) ** 2


def softplus2_inv(x):
    return np.log(np.exp(np.sqrt(x)) - 1.)


def softplus_inv(x):
    return np.log(np.exp(x) - 1.)
