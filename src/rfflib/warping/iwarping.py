import numpy as np
import torch
from torch import trapz


class InputWarping:
    def __init__(self):
        pass


class BetaCDF:
    """

    :return:
    """

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta


class SSWIM(InputWarping):
    def __init__(self):
        pass


class SSWIM_GX(SSWIM):
    def __init__(self):
        pass


class SSWIM_XGX(SSWIM):
    def __init__(self):
        pass


def beta_fn(alpha, beta):
    return (torch.exp(torch.lgamma(alpha)) * torch.exp(torch.lgamma(beta))) / torch.exp(torch.lgamma(alpha + beta))


def beta_inc(X, rand_samples, alpha, beta, dim_sum=0):
    """

    :param alpha:
    :param beta:
    :param X:   (N,D)
    :param rand_samples:
    :param dim_sum:
    :return:
    """
    rand_samples = X * rand_samples  # TODO CHECK THIS BROADCASTING
    rand_samples, _ = torch.sort(rand_samples, dim=0)
    y = rand_samples ** (alpha - 1) * (1 - rand_samples) ** (beta - 1)
    return torch.trapz(y, x=rand_samples, dim=dim_sum)


def beta_rinc(X, rand_samples, alpha=torch.tensor(0.5), beta=torch.tensor(0.5), dim_sum=0):
    """
    Regularised Incomplete Beta Function
    https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
    :param X:
    :param rand_samples:
    :param alpha:
    :param beta:
    :param dim_sum:
    :return:
    """
    denom = beta_fn(alpha, beta)
    return beta_inc(X, rand_samples, alpha, beta, dim_sum=dim_sum) / denom.view((1, denom.shape[-1]))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N_samples = 4000
    rand_samples = torch.tensor(np.random.uniform(0.0, 1.0, (N_samples, 1, 1)))
    N = 2000
    D = 1
    X = torch.tensor(np.random.uniform(0.0, 1.0, (2000, D)))
    # BCDF = BetaCDF(alpha=torch.tensor(0.5), beta = torch.tensor(0.5))
    alpha = torch.tensor(np.array([[0.4]])).view(1, 1, D)
    beta = torch.tensor(np.array([[3.0]])).view(1, 1, D)
    X_warped = beta_rinc(X, rand_samples, alpha, beta, dim_sum=0)

    plt.figure(dpi=200)
    ax = plt.subplot(2, 2, 1)
    plt.scatter(X[:, 0], X_warped[:, 0])
    plt.show()
