import numpy as np


def chirp_fn(t):
    return np.sin(2 * np.pi * 0.3 * (t + 1) ** 4)


def step_fn(t):
    return np.sign(t)


def step_2d_fn(t):
    return np.sign(t[:, 1])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Don't remove

    fig = plt.figure()
    x = np.linspace(-1, 1, 1000)

    ax = fig.add_subplot(221)
    y = chirp_fn(x)
    ax.plot(x, y)

    ax = fig.add_subplot(222)
    y = step_fn(x)
    ax.plot(x, y)

    ax = fig.add_subplot(212, projection='3d')
    xx, yy = np.meshgrid(x, x)
    xy = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    zz = step_2d_fn(xy).reshape(xx.shape)
    ax.plot_surface(xx, yy, zz)
    plt.show()
