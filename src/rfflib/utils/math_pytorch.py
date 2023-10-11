import numpy as np
import torch
from rfflib.utils.cholesky import cholesky_solve
from torch.nn.functional import softplus
from time import time

def asinh(x):
    return torch.log(x + (x ** 2 + 1) ** 0.5)


def acosh(x):
    return torch.log(x + (x ** 2 - 1) ** 0.5)


def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


def logit(x):
    """
    This is equivalent to the inverse of nn.functional.sigmoid()
    :param x:
    :return:
    """
    return torch.log(x / (1 - x))


def sigmoid_inv(x):
    return logit(x)


def softplus2(x, exponent=2):
    return softplus(x) ** exponent


def softplus2_inv(x, exponent=2):
    return torch.log(torch.exp(torch.pow(x, 1 / exponent)) - 1.)


# def softplus2(x):
#     return softplus(x)
#
#
# def softplus2_inv(x):
#     return softplus_inv(x)

def softplus_inv(x):
    return torch.log(torch.exp(x) - 1.)


def diff(a, axis=0):
    """
    equivalent of np.diff with n=1
    magic from: https://stackoverflow.com/a/42612608
    :param a:
    :param axis:
    :return:
    """
    if axis == 0:
        return a[1:] - a[:-1]
    elif axis == 1:
        return a[:, 1:] - a[:, :-1]


def mnlp(actual_mean, pred_mean, pred_var):
    """
    Mean Negative Log Probability
    :param actual_mean:
    :param pred_mean:
    :param pred_var:
    :return:
    """
    # log_part = torch.log(2 * np.pi * pred_var) / 2.0
    # unc_part = (pred_mean - actual_mean)**2 / (2 * pred_var)
    # summed_parts = log_part + unc_part
    # _mnlp = torch.sum(summed_parts)
    # return _mnlp
    log_part = torch.log(pred_var) + torch.log(2 * torch.tensor(np.pi))
    unc_part = ((actual_mean - pred_mean) / torch.sqrt(pred_var))**2
    summed_parts = 0.5 * (log_part + unc_part)
    _mnll = torch.sum(summed_parts)
    return _mnll


def nlml_smw(y_pred,
             mu,
             N,
             M,
             y_trn,
             alpha,
             beta,
             logdet):
    """
    Sherman-Morrison-Woodbury NLML
    NOTE: This is the loss as defined in Bishop
    :param Y_pred:
    :param mu:
    :param N:
    :param M:
    :param Y_trn:
    :param alpha:
    :param beta:
    :param logdet:
    :param tfdt:
    :return:
    """
    # y_hat = torch.mean(y_pred, dim=1, keepdim=True)
    p1 = (M / 2) * torch.log(alpha)
    p2 = (N / 2) * torch.log(beta)
    E_mn1 = (beta / 2.0) * torch.norm(y_trn - y_pred, p="fro", keepdim=False) ** 2
    # E_mn1 = (beta / 2.0) * torch.norm(y_trn - y_hat, p=1, keepdim=False) #** 2
    E_mn2 = (alpha / 2.0) * torch.matmul(mu.t(), mu)
    p4 = (1 / 2.0) * logdet
    p5 = (N / 2.0) * np.log(2 * np.pi)
    nlml = -1.0 * (p1 + p2 - E_mn1 - E_mn2 - p4 - p5)
    return nlml


def nlml_chol_full(y_trn,
                   PHI,
                   A,
                   mu,
                   n,
                   m,
                   alpha,
                   beta):
    p1 = torch.matmul(y_trn.t(), y_trn)
    p2 = torch.chain_matmul(y_trn.t(), PHI, mu)
    p3 = (1.0 / 2.0) * torch.logdet(A)
    p4 = (m / 2) * torch.log(alpha / beta)
    p5 = (n / 2) * torch.log((2 * np.pi) / beta)

    nlml_chol = -1.0 * (-(beta / 2) * (p1 - p2) - p3 + p4 - p5)
    return nlml_chol


def nlml_chol_fast(y_trn,
                   r,
                   rb_solve,
                   n,
                   m,
                   alpha,
                   beta):
    """
    Fast cholesky version
    :param y_trn:
    :param R:
    :param Rb_solve:
    :param N:
    :param M:
    :param alpha:
    :param beta:
    :param pi:
    :param tfdt:
    :return:
    """
    # tic = time()
    e1_chol = torch.norm(y_trn, p="fro", keepdim=False, ) ** 2
    e2_chol = torch.norm(rb_solve, p="fro", keepdim=False) ** 2
    e_chol = (- beta / 2.0) * (e1_chol - e2_chol)

    logdet_chol = (1.0 / 2.0) * torch.sum(torch.log(torch.diagonal(r) ** 2))
    p1_chol = (m / 2.0) * torch.log(alpha / beta)
    p2_chol = (n / 2.0) * torch.log((2.0 * np.pi) / beta)
    nlml_chol = -1.0 * (e_chol - logdet_chol + p1_chol - p2_chol)
    # toc = time()
    return nlml_chol


def smw_inv_precomp(a_inv, vs, r_lower):
    """
    Faster iterative SMW using cholesky solve solve for inversion
    NOTE: UPDATES A_inv (which is S)
    :param a_inv:
    :param vs:
    :param r_lower:
    :return:
    """
    solv1, _ = torch.solve(torch.eye(r_lower.shape[0]), r_lower)  # XXX TODO XXX replace this with torch.solve() in newer pytorch
    i_plus_vsu_inv, _ = torch.solve(solv1, r_lower.t())  # XXX TODO XXX replace this with torch.solve() in newer pytorch
    return a_inv - torch.chain_matmul(vs.t(), i_plus_vsu_inv, vs)


def logdet_correction_precomp(a_logdet, r_lower):
    """
    Fast iterative logdet using cholesky decomposed A
    :param s_logdet:
    :param r_lower:
    :return:
    """
    # NOTE: sign_det should be always +1 because SPD
    logdet_chol = (1.0 / 2.0) * torch.sum(torch.log(torch.diag(r_lower) ** 2), dim=0, keepdim=False)
    return a_logdet + logdet_chol


if __name__ == "__main__":
    from time import time
    from sklearn.decomposition import PCA as PCAsklearn, IncrementalPCA

    N = 100
    D = 100000
    X_numpy = np.random.normal(size=(N, D))
    X = torch.tensor(X_numpy)
    k = 100
    skl_pca = IncrementalPCA(n_components=k)  # , svd_solver="auto")
    t1 = time()
    skl_pca.fit(X_numpy)
    X_skl_pca = skl_pca.transform(X_numpy)
    t2 = time()
    print("time for sklearn pca: {}".format(np.round(t2 - t1, 3)))
    # my_pca = PCA(k=k, method="svd", center=True, scale=False)
    # my_pca.fit(X)
    # my_pca.transform(X)
