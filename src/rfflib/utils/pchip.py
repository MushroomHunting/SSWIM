import numpy as np
import torch
from rfflib.utils.math_pytorch import diff


def pchiptx(x, y, x_query):
    """

    :param x: The x value of the interpolating points
    :param y: The y value of the interpolating points
    :param x_query: The points we want to interpolate
    :return:

    """

    # PCHIPTX  Textbook piecewise cubic Hermite interpolation.
    #  v = pchiptx(x,y,u) finds the shape-preserving piecewise cubic
    #  interpolant P(x), with P(x(j)) = y(j), and returns v(k) = P(u(k)).

    #  First derivatives
    h = diff(a=x)
    delta = diff(a=y) / h
    d = pchipslopes(h, delta)

    #  Piecewise polynomial coefficients
    n, D = x.shape
    c = (3 * delta - 2 * d[0:n - 1] - d[1:n]) / h
    b = (d[0:n - 1] - 2 * delta + d[1:n]) / (h ** 2)

    #  Find subinterval indices k so that x(k) <= u < x(k+1)
    j_range = torch.arange(1, n - 1)  # .view(-1,1) #* torch.ones(size=(n-2,D),dtype= torch.int64)  # .reshape(-1, 1)
    K, _ = torch.max(((x[j_range] <= x_query[:, None, :]).long() * j_range.view(-1, 1)[None, :, :]), dim=1, keepdim=False)

    #  Evaluate interpolant
    # v = y[K] + s * (d[K] + s * (c[K] + s * b[K]))
    xK = torch.cat([x[:, i][K[:, i]].view(-1, 1) for i in range(D)], dim=1)
    yK = torch.cat([y[:, i][K[:, i]].view(-1, 1) for i in range(D)], dim=1)
    dK = torch.cat([d[:, i][K[:, i]].view(-1, 1) for i in range(D)], dim=1)
    cK = torch.cat([c[:, i][K[:, i]].view(-1, 1) for i in range(D)], dim=1)
    bK = torch.cat([b[:, i][K[:, i]].view(-1, 1) for i in range(D)], dim=1)
    s = x_query - xK

    v = yK + s * (dK + s * (cK + s * bK))

    return v, d


def pchipslopes(h, delta):
    """

    :param h: The first differences
    :param delta:
    :return:
    """
    #  PCHIPSLOPES  Slopes for shape-preserving Hermite cubic
    #  pchipslopes(h,delta) computes d(k) = P'(x(k)).

    #  Slopes at interior points
    #  delta = diff(y)./diff(x).
    #  d(k) = 0 if delta(k-1) and delta(k) have opposites
    #         signs or either is zero.
    #  d(k) = weighted harmonic mean of delta(k-1) and
    #         delta(k) if they have the same sign.
    n, D = h.shape
    d = torch.zeros(size=(n + 1, D))

    """
    val_to_check = torch.sign(delta[0:n - 1]) * torch.sign(delta[1:n])
    # k = np.argwhere(val_to_check > 0).flatten() + 1  # XXX TODO XXX DO WE NEED +1 for python ??!?!
    k = ((val_to_check != 0).nonzero()) + 1  # XXX TODO XXX DO WE NEED +1 for python ??!?!
    w1 = 2 * h[k] + h[k - 1]
    w2 = h[k] + 2 * h[k - 1]
    d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    """

    val_to_check = torch.sign(delta[0:n - 1]) * torch.sign(delta[1:n])
    ks = [((val_to_check[:, d] != 0).nonzero().view(-1)) + 1 for d in range(D)]
    w1 = 2 * torch.cat([h[:, i][k].view(-1, 1) for i, k in enumerate(ks)], dim=1) \
         + torch.cat([h[:, i][k - 1].view(-1, 1) for i, k in enumerate(ks)], dim=1)
    w2 = torch.cat([h[:, i][k].view(-1, 1) for i, k in enumerate(ks)], dim=1) \
         + 2 * torch.cat([h[:, i][k - 1].view(-1, 1) for i, k in enumerate(ks)], dim=1)
    for i, k in enumerate(ks):
        d[:, i][k] = (w1[:, i] + w2[:, i]) / (w1[:, i] / delta[:, i][k - 1] + w2[:, i] / delta[:, i][k])

    #  Slopes at endpoints
    d[0] = pchipend(h[0], h[1], delta[0], delta[1])
    d[n] = pchipend(h[n - 1], h[n - 2], delta[n - 1], delta[n - 2])
    return d


def pchipend(h1, h2, del1, del2):
    """

    :param h1:
    :param h2:
    :param del1:
    :param del2:
    :return:
    """
    #  Noncentered, shape-preserving, three-point formula.
    d = ((2 * h1 + h2) * del1 - h1 * del2) / (h1 + h2)
    cond1 = torch.sign(d) != torch.sign(del1)
    cond2 = (torch.sign(del1) != torch.sign(del2)) * (abs(d) > abs(3 * del1))
    cond3 = ~cond1 * ~cond2
    # return cond1 * torch.zeros_like(d) \
    #            + cond2 * 3 * del1 \
    #            + cond3 * d
    return cond1.type(torch.get_default_dtype()) * torch.zeros_like(d) \
           + cond2.type(torch.get_default_dtype()) * 3 * del1 \
           + cond3.type(torch.get_default_dtype()) * d


if __name__ == "__main__":
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
    plt.figure(dpi=200)
    x_raw = np.array([0.001, 0.13139251, 0.68542963, 0.75127256, 0.999])
    y_raw = np.array([-100., -60.302124, 27.84745, 77.6573, 100.])
    x = torch.from_numpy(x_raw)
    y = torch.from_numpy(y_raw)
    # y = -1*np.array([ 12.0 ,3.0, 0.9, 0.9, 0.1,  ])
    # y = np.random.uniform(-1,1, size=(x.shape))
    x_query = torch.from_numpy(np.linspace(0.0, 1.0, 1000))
    y_query, derivs = pchiptx(x=x, y=y, x_query=x_query)
    derivs = derivs.data.numpy()
    x_endderiv = np.linspace(x_raw[-1], x_raw[-1] + 0.3)
    x_startderiv = np.linspace(x_raw[0] - 0.3, x_raw[0])
    y_enderiv = y_raw[-1] + derivs[-1] * x_endderiv
    y_startderiv = y_raw[0] + derivs[0] * x_startderiv

    plt.plot(x_query.data.numpy(), y_query.data.numpy())
    plt.plot(x_endderiv, y_enderiv, c="red")
    plt.plot(x_startderiv, y_startderiv, c="red")
    # print("f2: {}".format(f2))
    # print("y_query: {}".format(y_query))
    plt.show()
