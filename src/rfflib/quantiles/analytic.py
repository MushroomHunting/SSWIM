import scipy.stats as sstats
import numpy as np
import torch
import torch.distributions as tdists


def norm(points, loc, scale):
    """
    Standard Normal
    :param points:
    :param loc:     mean
    :param scale:   standard deviation
    :return:
    """
    _loc = torch.tensor(loc)
    _scale = torch.tensor(scale)
    p = tdists.Normal(loc=_loc, scale=_scale)
    return p.icdf(points)


def normal(points, ls, meanshift=0):
    """
    Standard Normal
    :param points:
    :param meanshift: an optional shifting of the distribution's mean (e.g. for lebesgue-stieltjes features)
    :return:
    """
    _loc = torch.tensor(0.0)
    _scale = torch.tensor(1.0)
    p = tdists.Normal(loc=_loc, scale=_scale)
    return meanshift + p.icdf(points)/ls


def matern_12(points, ls, meanshift=0):
    """
    Matern 12
    Ref: Sampling Student''s T distribution-use of the inverse cumulative distribution function
    :param points:
    :param meanshift: an optional shifting of the distribution's mean (e.g. for lebesgue-stieltjes features)
    :return:
    """
    return meanshift + torch.tan(np.pi * (points - 0.5))/ ls


def matern_32(points, ls, meanshift=0):
    """
    Matern 32
    ref: Ref: Sampling Student''s T distribution-use of the inverse cumulative distribution function
    :param points:
    :param meanshift: an optional shifting of the distribution's mean (e.g. for lebesgue-stieltjes features)
    :return:
    """
    return meanshift + ((2 * points - 1) / torch.sqrt(2 * points * (1 - points))) / ls


def matern_52(points, ls, meanshift=0):
    """
    Matern 52
    Ref: Sampling Student''s T distribution-use of the inverse cumulative distribution function
    :param points:
    :param meanshift: an optional shifting of the distribution's mean (e.g. for lebesgue-stieltjes features)
    :return:
    """
    alpha = 4 * points * (1 - points)
    p = 4 / torch.sqrt(alpha) * torch.cos((1 / 3) * torch.acos(torch.sqrt(alpha)))
    return meanshift + (torch.sign(points - 0.5) * torch.sqrt(p - 4)) /  ls
