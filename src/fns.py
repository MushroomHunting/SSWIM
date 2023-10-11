import numpy as np


# Generate fake data:
def step(X):
    """
    Single step function.
    f(x) = 0        if x < 0
    f(x) = 0.5      if x == 0
    f(x) = 1        if x > 0
    :param X:
    :return:
    """
    y_1 =  0.5 * (np.sign(X-0.5) ) + X
    y_1_greatermask = (y_1 > 0.5)
    y_1[y_1_greatermask] += 2
    # y_1_greatermask = (y_1 > 6.5)
    # y_1[y_1_greatermask] -= 3
    return y_1


def chirpy1(X):
    """
    Single step function.
    f(x) = 0        if x < 0
    f(x) = 0.5      if x == 0
    f(x) = 1        if x > 0
    :param X:
    :return:
    """
    return np.sin(((X - 0.5) ** 2 ) * 12) * X + (X) ** 2
    # return np.sin(((X - 1.5) ** 2 ) * 11) * X + (X) ** 3


# Generate fake data:
def flatcorners2(X):
    # return 20 * (np.abs(-2 + np.cos(s * 4.0 + a * 16 + 0.1) + s * 2.0))**2 - 50
    retval = (X[:, 0] - 1.5) * (X[:, 1] - 1.5)
    return retval.reshape(-1, 1)


def shubert(X):
    """
    Search domain: −10 ≤ xi ≤ 10, i = 1, 2.
    Number of local minima: several local minima.
    The global minima: 18 global minima  f(x*) = -186.7309.
    http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page1882.htm
    :param X:
    :return:
    """
    D = X.shape[1]
    s1 = 0
    s2 = 0
    for i in range(5):
        s1 = s1 + (i + 1) * np.cos(((i + 1) + 1) * X[:, 0] + (i + 1))
        s2 = s2 + (i + 1) * np.cos(((i + 1) + 1) * X[:, 1] + (i + 1))
    retval = s1 * s2
    return retval.reshape(-1, 1)


def gramacy2dexp(X):
    """
    Search domain: [−2, 6] × [−2, 6].
    Number of local minima: several local minima.
    The global minima: 18 global minima  f(x*) = -186.7309.
    http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page1882.htm
    :param X:
    :return:
    """

    return (X[:, 0] * np.exp(-X[:, 0] ** 2 - X[:, 1] ** 2)).reshape(-1, 1)


def schwefel(X):
    """
    Search domain: −500 ≤ xi ≤ 500, i = 1, 2, . . . , n.
    at X=(420.9687,…,420.9687), f(X) = 0
    :param X: (N,D)
    :return:
    """
    D = X.shape[1]
    s = np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)
    retval = 418.9829 * D - s
    return retval.reshape(-1, 1)


def michalewicz(X):
    """
    search domain: [0, np.pi]
     The global minima:          at n=2, f(x*) = -1.8013
    http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2376.htm
    :param X:
    :return:
    """
    D = X.shape[1]

    m = 10
    s = 0
    for i in range(D):
        s = s + np.sin(X[:, i]) * (np.sin((i + 1) * X[:, i] ** 2 / np.pi)) ** (
                2 * m)

    retval = -s
    return retval.reshape(-1, 1)


def ackley(X):
    """
    Search domain: [-15, 30]
    The global minima:          at n=2, f(x*) = 0.0
    http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page295.htm
    :param X:
    :param D:
    :return:
    """
    D = X.shape[1]
    # X = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1)))
    a = 20
    b = 0.2
    c = 2 * np.pi
    s1 = np.zeros(shape=(X.shape[0], 1))
    s2 = np.zeros(shape=(X.shape[0], 1))
    for i in range(D):
        s1 = s1 + X[:, i].reshape(-1, 1) ** 2
        s2 = s2 + np.cos(c * X[:, i].reshape(-1, 1))
    retval = (-a * np.exp(-b * np.sqrt(1 / D * s1)) - np.exp(
        1 / D * s2) + a + np.exp(1)).flatten()
    return retval.reshape(-1, 1)


def rastrigin(X):
    D = X.shape[1]
    A = 10
    retval = (A * D) + np.sum(X ** 2 - A * np.cos(2 * np.pi * X), axis=1)
    return retval.reshape(-1, 1)


def hartmann6(X):
    """
    6d Hartmann test function
    constraints:
    0 <= xi <= 1, i = 1..6
    global optimum at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
    where har6 = -3.32236

    https://github.com/automl/HPOlib/blob/development/HPOlib/benchmarks/benchmark_functions.py
    """

    a = np.array([[10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                  [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                  [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                  [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]])
    c = np.array([1.0, 1.2, 3.0, 3.2])
    p = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                  [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                  [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                  [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    s = 0
    for i in [0, 1, 2, 3]:
        sm = a[i, 0] * (X[:, 0] - p[i, 0]) ** 2
        sm += a[i, 1] * (X[:, 1] - p[i, 1]) ** 2
        sm += a[i, 2] * (X[:, 2] - p[i, 2]) ** 2
        sm += a[i, 3] * (X[:, 3] - p[i, 3]) ** 2
        sm += a[i, 4] * (X[:, 4] - p[i, 4]) ** 2
        sm += a[i, 5] * (X[:, 5] - p[i, 5]) ** 2
        s += c[i] * np.exp(-sm)
    retval = -s
    return retval.reshape(-1, 1)


def rosenbrock(X):
    scores = 0
    D = X.shape[1]
    a = 1
    b = 100
    # for i = 1 : (n-1)
    for i in range(D - 1):
        scores += (b * ((X[:, i + 1] - (X[:, i] ** 2)) ** 2)) + (
                (a - X[:, i]) ** 2)
    retval = scores
    return retval.reshape(-1, 1)
