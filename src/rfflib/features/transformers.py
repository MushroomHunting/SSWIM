import math
import torch
from time import time


def linear(X, has_bias_term=True):
    """
    Allows a linear regression model to model a linear trend
    Optionally allows a bias term
    :param X: (N,D) tf.placeholder or tf.Variable or tf.Tensor
    :param has_bias_term:  boolean

    :return:
    """
    shape = X.shape
    if has_bias_term is True:
        return torch.cat((torch.ones(size=(shape[0], 1)), X), dim=1)
    else:
        return X


def cos_sin(X, S):
    """
    :param X:   (N,D) tensor
    :param S:   list
                List of frequencies sampled, potentially, ARD

    :return:
    """
    D, m = S.shape
    inside_calc = torch.matmul(X, S)
    return torch.cat((torch.cos(inside_calc), torch.sin(inside_calc)), dim=1) / math.sqrt(m)


def cos_sin_ns(X, S, X_var):
    """
    Nonstationary fourier features (lebesgue-steltjes)
    :param X: (N,D)
    :return:
    """
    S1, S2 = S
    D, m = S1.shape
    inside_calc1 = torch.matmul(X, S1)
    inside_calc2 = torch.matmul(X, S2)
    return torch.div(torch.cat((torch.cos(inside_calc1), torch.sin(inside_calc1)), dim=1) +
                     torch.cat((torch.cos(inside_calc2), torch.sin(inside_calc2)), dim=1),
                     math.sqrt(4 * m))


def cos_sin_ui(X, S, X_var):
    """
    Uncertain inputs prediction transformation
    :param X:
    :param X_var:
    :param S:
    :return:
    """
    if X_var is None:
        return cos_sin(X, S)
    else:
        # tic = time()
        D, m = S.shape
        inside_calc = torch.matmul(X, S)
        # IMPORTANT: This is only for diagonal covariance! Which is the case for us here.
        S2 = S * S  # (D,m)
        if X_var.shape[1] == 1:
            ui_exp = torch.matmul(X_var.repeat(1, D), S2)
        else:
            ui_exp = torch.matmul(X_var, S2)

        ci_phi = torch.cat((torch.cos(inside_calc), torch.sin(inside_calc)), dim=1) / math.sqrt(m)
        ui_phi = torch.exp(-ui_exp / 2.0)
        ui_phi = torch.cat((ui_phi, ui_phi), dim=1)
        # toc = time()
        # print(f"time taken: {toc-tic} s")
        return ui_phi * ci_phi
