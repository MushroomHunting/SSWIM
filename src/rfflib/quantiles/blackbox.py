from rfflib.utils.interpolate import pchiptx_asymp
import torch


def pchip_quantile(qpoints, sequence, dtype=torch.float32):
    """

    :param qpoints:     list
                        A list of qpoints for each dimension
    :param sequence:    array
                        A list of sequences on a per dimension basis
    :param is_nonstationary:
    :return:
    """
    new_qpoints_x, new_qpoints_y = qpoints.get_points()
    return pchiptx_asymp(x=new_qpoints_x,
                         y=new_qpoints_y,
                         x_query=sequence.points, dtype=dtype)
