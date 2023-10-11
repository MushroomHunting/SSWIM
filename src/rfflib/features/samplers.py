from rfflib.quantiles import analytic as analytic_quantiles, blackbox as blackbox_quantiles
from rfflib.utils.enums import STANDARD_KERNEL
import torch


def bbq_kernel(sequence,
               qpoints,
               interp_type="pchipasymp",
               ns_type="lebesgue_stieltjes"):
    """

    :param sequence:        Sequence
    :param qpoints:         list
                            List of Quantile Point Parametrisation object
                            Object containing quantile points
    :param interp_type:    function
                            the interpolating function
    :return:
    """

    if interp_type == "pchipasymp":
        if ns_type is None:
            S = blackbox_quantiles.pchip_quantile(qpoints, sequence)
            return S.t()
        elif ns_type == "lebesgue_stieltjes":
            S = [blackbox_quantiles.pchip_quantile(qpts, sequence) for qpts in qpoints]
            return [omega.t() for omega in S]


def standard_kernel(sequence, kernel_type=STANDARD_KERNEL.RBF, ls=1.0, ns_type=None, meanshift=None):
    """
    General wrapper for all standard Quasi-Monte Carlo (Fourier) Features
    :param sequence:     QMCSequence object
    :param kernel_type:  enum
    :param ns_type:      string or None
    :return:
    """
    quantiles_lut = {STANDARD_KERNEL.RBF: analytic_quantiles.normal,
                     STANDARD_KERNEL.M12: analytic_quantiles.matern_12,
                     STANDARD_KERNEL.M32: analytic_quantiles.matern_32,
                     STANDARD_KERNEL.M52: analytic_quantiles.matern_52}
    qf = quantiles_lut[kernel_type]
    # sequence.init_points()

    if ns_type is None:
        S = qf(sequence.points, ls)
        return S.t() #/ ls.forward().t()
    elif ns_type == "lebesgue_stieltjes":
        S = [qf(sequence.points, ls[i], meanshift[i]) for i in range(len(ls))]
        return [omega.t()  for omega in S]
