import numpy as np
# from torch.nn.functional import softplus
# from torch import sigmoid
import torch



# from torch import clamp
import matplotlib.pyplot as plt
# from rfflib.utils.math_numpy import sigmoid_inv as inverse_sigmoid
from rfflib.utils.opt import Param



class BoundedQPoints():
    def __init__(self,
                 n,
                 d,
                 rand_init=True,
                 x_minmax=(0.1, 0.90),
                 y1_minmax=(-1.0, 1.0),
                 y2_offset_max=1):

        """
        :param n:   integer
                    Number of interpolating points
        :param random_init: Boolean
        :param dtype:
        """
        self.n = n
        self.d = d
        self.x_minmax = x_minmax
        self.y1_minmax = y1_minmax
        self.y2_offset_max = y2_offset_max

        # 2D offset coordinates
        if rand_init:
            initial_x_left = np.random.uniform(self.x_minmax[0], self.x_minmax[0], size=(1, self.d))
            initial_x_right = np.random.uniform(self.x_minmax[1], self.x_minmax[1], size=(1, self.d))
            self.x_left = Param(initial_x_left)
            self.x_right = Param(initial_x_right)

            initial_offsets_x = np.random.uniform(0.0, 1.0, size=(self.n - 1, self.d))
            initial_offsets_y = np.random.uniform(0.0, 1.0, size=(self.n - 1, self.d))

            self.offsets_x = Param(initial_offsets_x)
            self.offsets_y = Param(initial_offsets_y)
            y1_init = np.random.uniform(self.y1_minmax[0], self.y1_minmax[1], size=(1, self.d))

            y2_offset_init = np.random.uniform(self.y2_offset_max, self.y2_offset_max, size=(1, self.d))

            self.y1 = Param(y1_init)
            self.y2_offset = Param(y2_offset_init)  # must be positive
            self.params = [self.y1, self.y2_offset, self.offsets_x, self.offsets_y, self.x_left, self.x_right]
            """
            e.g. 
            self.y1            : {-100, 100}           { unconstrained,  unconstrained }  
            self.y2_offset     : [0.0001, 200}         [ min 0, unconstrained}
            self.offsets_x     : [0.0001, 0.9999]      [constrained, constrained]
            self.offsets_y     : [0.0001, 0.9999]      [constrained, constrained]
            """

    def get_params(self):
        """
        Returns th
        :return:
        """
        return self.params

    def get_points(self):
        """

        :return:
        """
        points_x = torch.zeros(size=(self.n, self.d))
        points_y = torch.zeros(size=(self.n, self.d))

        p1x = self.x_left.forward()  # This assumes x_left is a matrix
        p2x = self.x_right.forward()

        p1y = self.y1.forward()

        delta_x = p2x - p1x
        delta_y = self.y2_offset.forward()

        points_x[:, :] = p1x
        points_y[:, :] = p1y

        offsets_x = self.offsets_x.forward()  # .float()  # (N,D)
        offsets_y = self.offsets_y.forward()  # .float()  # (N,D)

        offsets_x_sum = torch.sum(offsets_x, dim=0)
        offsets_y_sum = torch.sum(offsets_y, dim=0)

        rescale_x = offsets_x_sum / delta_x
        rescale_y = offsets_y_sum / delta_y

        # XXX TODO XXX IS THERE A WAY TO VECTORIZE THIS NEATLY??
        for i in range(1, self.n):
            points_x[i:, :] += offsets_x[i - 1, :] / rescale_x
            points_y[i:, :] += offsets_y[i - 1, :] / rescale_y

        return points_x, points_y


def visualise_qpoints(qpoints, dims_to_plot=None, N_query=20000):
    """

    :param qpoints:
    :param dims_to_plot:    list or None
                            if list, then plot the given indexes; e.g. [0,1,3,4]
    :param N_query:
    :return:
    """
    # from utils.pchip import pchiptx
    from rfflib.utils.interpolate import pchiptx_asymp
    a = qpoints
    D = a.d
    my_points_torch_x, my_points_torch_y = a.get_points()
    my_points_x = my_points_torch_x.data.numpy()
    my_points_y = my_points_torch_y.data.numpy()

    x_query = np.random.uniform(0, 1, size=(N_query, D))
    x_query_torch = torch.tensor(x_query)
    # y_query_torch, derivs_torch = pchiptx(x=my_points_torch_x,
    #                                       y=my_points_torch_y,
    #                                       x_query=x_query_torch,
    #                                       dtype=dtype)
    # y_query = y_query_torch.data.numpy()
    y_query_torch_asymp = pchiptx_asymp(x=my_points_torch_x,
                                        y=my_points_torch_y,
                                        x_query=x_query_torch)
    y_asymp_query = y_query_torch_asymp.data.numpy()
    plt.figure(dpi=200, figsize=(12, 9))

    if dims_to_plot is None:
        dims_to_plot = np.arange(D)

    for d in dims_to_plot:
        ax = plt.subplot(3, D, d + 1)
        ax.scatter(my_points_x[:, d], my_points_y[:, d], s=10)
        ax.plot(my_points_x[:, d], my_points_y[:, d])  # , s=10)
        ax.set_title("raw points, dim {}".format(d))
        ax.set_xlim(-0.1, 1.1)

        # ax = plt.subplot(3, D, d + 1 + 1 * D)  # plot on 2nd row
        # ax.scatter(my_points_x[:, d], my_points_y[:, d], s=10)
        sort_idxs = np.argsort(x_query[:, d])
        # ax.plot(x_query[sort_idxs, d], y_query[sort_idxs, d], c="r")  # , s=10)
        # ax.set_title("Raw points, dim {}".format(d))

        ax = plt.subplot(3, D, d + 1 + 1 * D)  # plot on 3rd row
        ax.scatter(my_points_x[:, d], my_points_y[:, d], s=10)
        ax.plot(x_query[sort_idxs, d], y_asymp_query[sort_idxs, d], c="r")  # , s=10)
        ax.set_title("pchiptx_asymp, dim {}".format(d))
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-50, 50)  # XXX TODO XXX make this change depending on the object y min max attribute values

    plt.tight_layout()


if __name__ == "__main__":
    # use_gpu = True
    use_gpu = False
    # dtype = "f32"
    dtype = "f64"

    if dtype == "f32":
        dtype_np = np.float32
        if use_gpu:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
    elif dtype == "f64":
        dtype_np = np.float64
        if use_gpu:
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            torch.set_default_tensor_type(torch.DoubleTensor)


    from time import time
    import matplotlib
    import matplotlib.pylab as plt
    import seaborn as sbn
    from rfflib.utils.pchip import pchiptx
    from rfflib.utils.interpolate import pchiptx_asymp
    # from rfflib.utils.math_numpy import inverse_sigmoid

    matplotlib.rcParams.update(
        {'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42,
         'legend.fontsize': 8, 'image.cmap': "viridis"})
    sbn.set(font_scale=0.4)
    sbn.set_context(rc={'lines.markeredgewidth': 0.25})
    sbn.set_style("whitegrid")

    print("Hi")
    # np.random.seed(1)
    D = 5
    N_points = 30  # 35
    N_query = 20000
    bqp = BoundedQPoints(n=N_points,
                         d=D,
                         rand_init=True)
    my_points_torch_x, my_points_torch_y = bqp.get_points()
    my_points_x = my_points_torch_x.data.numpy()
    my_points_y = my_points_torch_y.data.numpy()

    # np.random.seed(37)
    x_query = np.random.uniform(0, 1, size=(N_query, D))
    x_query_torch = torch.tensor(x_query)  # linspace(x_raw[0], x_raw[-1], 1000))
    t1 = time()
    y_query_torch, derivs_torch = pchiptx(x=my_points_torch_x,
                                          y=my_points_torch_y,
                                          x_query=x_query_torch)
    t2 = time()
    time_taken = t2 - t1
    print("time for pchiptx: {}s,       D={}, N_interp={}, N_query={}".format(np.round(time_taken, 5), D, N_points, N_query))
    y_query = y_query_torch.data.numpy()

    t1 = time()
    y_query_torch_asymp = pchiptx_asymp(x=my_points_torch_x,
                                        y=my_points_torch_y,
                                        x_query=x_query_torch)
    t2 = time()
    time_taken = t2 - t1
    print("time for pchip_asymp: {}s,    D={}, N_interp={}, N_query: {}".format(np.round(time_taken, 5), D, N_points, N_query))
    y_asymp_query = y_query_torch_asymp.data.numpy()

    from scipy.interpolate import pchip_interpolate

    plt.figure(dpi=200, figsize=(12, 9))
    for d in range(D):
        ax = plt.subplot(3, D, d + 1)
        ax.scatter(my_points_x[:, d], my_points_y[:, d], s=10)
        ax.plot(my_points_x[:, d], my_points_y[:, d])  # , s=10)

        ax = plt.subplot(3, D, d + 1 + 1 * D)  # plot on 2nd row
        ax.scatter(my_points_x[:, d], my_points_y[:, d], s=10)
        sort_idxs = np.argsort(x_query[:, d])
        ax.plot(x_query[sort_idxs, d], y_query[sort_idxs, d], c="r", label="pytorch")  # , s=10)

        spv = pchip_interpolate(my_points_x[:, d], yi=my_points_y[:, d], x=x_query[sort_idxs, d])
        ax.plot(x_query[sort_idxs, d], spv, c="b", ls="--", label="scipy")  # , s=10)
        plt.legend()

        ax = plt.subplot(3, D, d + 1 + 2 * D)  # plot on 3rd row
        ax.scatter(my_points_x[:, d], my_points_y[:, d], s=10)
        ax.plot(x_query[sort_idxs, d], y_asymp_query[sort_idxs, d], c="r")  # , s=10)
        ax.set_ylim(-125, 125)

    plt.show()
