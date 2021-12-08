from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

headers = PrettyTable(['№', 'x1', 'x2', 'f(x1,x2)', 'df'])


def function(x, y):
    return 22 * x + 0.1 * y + np.exp(4.84 * x * x + 1.2 * y * y)


def extremum(a, b):
    return (a + b) / 2


def table(n, x1, x2, new_function, old_function):
    table_list = {
        'n': n,
        'x1': round(x1, 8),
        'x2': round(x2, 8),
        'f(x1,x2)': round(new_function, 10),
        'df': round(abs(new_function - old_function), 10)
    }
    headers.add_row(table_list.values())


def gss_x1(a, b, x2, eps):
    interval = b - a
    s = (sqrt(5) - 1) / 2
    a1 = a + interval * (1 - s)
    b1 = a + interval * s
    fa1 = function(a1, x2)
    fb1 = function(b1, x2)
    while abs(b - a) >= eps:
        if fa1 <= fb1:  # <= - минимум, >= - максимум
            b = b1
            b1 = a1
            fb1 = fa1
            interval = interval * s
            a1 = a + interval * (1 - s)
            fa1 = function(a1, x2)
        else:
            a = a1
            a1 = b1
            fa1 = fb1
            interval = interval * s
            b1 = a + interval * s
            fb1 = function(b1, x2)
    return extremum(a, b)


def gss_x2(c, d, x1, eps):
    interval = d - c
    s = (sqrt(5) - 1) / 2
    c1 = c + interval * (1 - s)
    d1 = c + interval * s
    fc1 = function(x1, c1)
    fd1 = function(x1, d1)
    while abs(d - c) >= eps:
        if fc1 <= fd1:  # <= - минимум, >= - максимум
            d = d1
            d1 = c1
            fd1 = fc1
            interval = interval * s
            c1 = c + interval * (1 - s)
            fc1 = function(x1, c1)
        else:
            c = c1
            c1 = d1
            fc1 = fd1
            interval = interval * s
            d1 = c + interval * s
            fd1 = function(x1, d1)
    return extremum(c, d)


def result(eps, x1, x2, new_function):
    return f'eps = {eps}\nx1 = {x1}, x2 = {x2}\nf(x1,x2) = {new_function}'


def coordinate_search(x1, x2, dx, eps):
    points_x = []
    points_y = []
    func = []
    count = 0
    k = 5
    eps_y = k * eps
    new_function = function(x1, x2)
    old_function = new_function + 1
    points_x.append(x1)
    points_y.append(x2)
    while abs(new_function - old_function) > eps_y:
        old_function = new_function
        count += 1
        a, c = x1 - dx, x2 - dx
        b, d = x1 + dx, x2 + dx

        x1 = gss_x1(a, b, x2, eps)
        new_function = function(x1, x2)
        points_x.append(x1)
        points_y.append(x2)
        func.append(function(points_x[-1], points_y[-1]))
        table(count, x1, x2, new_function, old_function)
        x2 = gss_x2(c, d, x1, eps)
        new_function = function(x1, x2)
        table('-', x1, x2, new_function, old_function)
        points_x.append(x1)
        points_y.append(x2)
        func.append(function(points_x[-1], points_y[-1]))
    return result(eps, x1, x2, new_function), [points_x, points_y], func


if __name__ == '__main__':
    x0 = y0 = -1
    dx = 2
    epsilon = 0.000000001
    info, points_coord, coord_func = coordinate_search(x0, y0, dx, epsilon)
    print(headers)
    print(info)

    x_axis = y_axis = np.arange(-1 * 1.1, 1 * 1.1, 0.0005)

    X, Y = np.meshgrid(x_axis, y_axis)
    Zs = np.array(function(np.ravel(X), np.ravel(Y)))
    Z = Zs.reshape(X.shape)
    coord_func.sort()
    cs = plt.contour(X, Y, Z, levels=coord_func)
    plt.clabel(cs)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.plot(points_coord[0], points_coord[1])

    plt.show()
