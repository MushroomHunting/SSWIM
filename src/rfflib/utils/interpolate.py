from rfflib.utils.math_pytorch import diff
from rfflib.utils.pchip import pchiptx
import torch


def pchiptx_asymp(x, y, x_query, dtype=torch.float64):
    """

    :param x:       (N,D)
                    All the interpolation "training" x values
    :param y:       (N,D)
                    All the interpolation "training" y values

    :param x_query: (N,D)
                    All the interpolation query values
    :return:
    """
    L_x = x[0]
    L_y = y[0]
    R_x = x[-1]
    R_y = y[-1]

    left_mask = x_query < L_x
    # middle_mask = (x_query >= L_x) & (x_query <= R_x)
    right_mask = x_query > R_x

    """
    query_R = x_query[right_mask]
    query_L = x_query[left_mask]

    all_interp, d = pchiptx(x, y, x_query)  # d is orignally (N_query,), now it's (N_query,D)
    middle_interp = all_interp[middle_mask]

    deriv_R = d[-1]
    deriv_L = d[0]

    a_R = deriv_R * (R_x - 1) ** 2
    a_L = deriv_L * L_x ** 2
    right_interp = - a_R / (query_R - 1.0) + a_R / (R_x - 1) + R_y
    left_interp = - a_L / query_L + a_L / L_x + L_y

    return torch.cat((left_interp, middle_interp, right_interp), dim=0).view(x_query.shape) 
    """

    n, D = x.shape
    all_interp, d = pchiptx(x, y, x_query)  # d is orignally (N_query,), now it's (N_query,D)

    for i in range(D):
        query_R = x_query[right_mask[:, i], i]
        query_L = x_query[left_mask[:, i], i]
        deriv_R = d[:, i][-1]
        deriv_L = d[:, i][0]
        a_R = deriv_R * (R_x[i] - 1) ** 2
        a_L = deriv_L * L_x[i] ** 2
        right_interp = - a_R / (query_R - 1.0) + a_R / (R_x[i] - 1) + R_y[i]
        left_interp = - a_L / query_L + a_L / L_x[i] + L_y[i]
        all_interp[right_mask[:, i], i] = right_interp
        all_interp[left_mask[:, i], i] = left_interp

    return all_interp


if __name__ == "__main__":
    import numpy as np
    import matplotlib
    import matplotlib.pylab as plt
    import seaborn as sbn

    matplotlib.rcParams.update(
        {'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42,
         'legend.fontsize': 8, 'image.cmap': "viridis"})
    sbn.set(font_scale=0.4)
    sbn.set_context(rc={'lines.markeredgewidth': 0.25})
    sbn.set_style("whitegrid")

    np.random.seed(37)
    # # h = np.array([0, 2, 2, 14, 10])
    # # delta = np.array([3, 3, 4, 10, 13])
    # # f2 = pchipslopes(h=h, delta=delta)
    #
    # x = np.array([1., 2., 4., 8., 9.])
    # y = np.array([0.1, 0.9, 0.9, 3.0, 12.0])
    # # y = -1*np.array([ 12.0 ,3.0, 0.9, 0.9, 0.1,  ])
    # # y = np.random.uniform(-1,1, size=(x.shape))
    # x_query = np.linspace(1.0, 9.0, 7)
    # y_query = pchiptx(x=x, y=y, x_query=x_query)
    #
    # plt.plot(x_query, y_query)
    # # print("f2: {}".format(f2))
    # print("y_query: {}".format(y_query))
    # plt.show()

    # EXAMPLE OF BREAKAGE
    # h = np.array([0, 2, 2, 14, 10])
    # delta = np.array([3, 3, 4, 10, 13])
    # f2 = pchipslopes(h=h, delta=delta)

    plt.figure(dpi=200)
    x_raw = np.array([0.1, 0.13139251, 0.68542963, 0.75127256, 0.8])
    y_raw = np.array([-100., -60.302124, 27.84745, 77.6573, 100.])
    x = torch.from_numpy(x_raw)
    y = torch.from_numpy(y_raw)
    # y = -1*np.array([ 12.0 ,3.0, 0.9, 0.9, 0.1,  ])
    # y = np.random.uniform(-1,1, size=(x.shape))
    x_query = torch.from_numpy(np.linspace(x_raw[0], x_raw[-1], 1000))
    y_query, derivs = pchiptx(x=x, y=y, x_query=x_query)
    derivs = derivs.data.numpy()
    x_endderiv = np.linspace(x_raw[-1], x_raw[-1] + 0.3)
    x_startderiv = np.linspace(x_raw[0] - 0.3, x_raw[0])
    b_enderiv = y_raw[-1] - derivs[-1] * x_raw[-1]
    b_startderiv = y_raw[0] - derivs[0] * x_raw[0]
    y_enderiv = b_enderiv + derivs[-1] * x_endderiv
    y_startderiv = b_startderiv + derivs[0] * x_startderiv

    x_query_fullspan_raw = np.linspace(0.0, 1.0, 1000)
    x_query_fullspan = torch.from_numpy(x_query_fullspan_raw)
    y_query_fullspan = pchiptx_asymp(x=x, y=y, x_query=x_query_fullspan)

    plt.plot(x_query_fullspan_raw, y_query_fullspan.data.numpy(), c="green", label="pchip + asymptotes")
    plt.plot(x_query.data.numpy(), y_query.data.numpy(), c="blue")
    plt.scatter(x_raw, y_raw, s=10)

    plt.ylim(-1000, 1000)

    plt.plot(x_endderiv, y_enderiv, c="red", label="endpoint derivatives")
    plt.plot(x_startderiv, y_startderiv, c="red")
    # print("f2: {}".format(f2))
    # print("y_query: {}".format(y_query))
    plt.legend()
    plt.show()
