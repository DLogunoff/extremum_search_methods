from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

headers = PrettyTable(
    ['№', 'x1_1', 'x2_1', 'x1_2', 'x2_2', 'function(x1,x2)', 'df'])


def table(count, old_function, x1_1, x2_1, x1_2, x2_2, new_function):
    Tablelist = {
        '№': count,
        'x1_1': round(x1_1, 7),
        'x2_1': round(x2_1, 7),
        'x1_2': round(x1_2, 7),
        'x2_2': round(x2_2, 7),
        'function(x1,x2)': round(new_function, 8),
        'df': abs(round(new_function - old_function, 8)),
    }
    headers.add_row(Tablelist.values())


def output(x1, x2, count, eps_y):
    return (f'Число шагов = {count}\nx1 = {x1}, x2 = {x2}\n'
            f'function(x1,x2) = {function(x1, x2)}\neps_y = {eps_y}')


def function(x1, x2):
    #return 10 * x1 * x1 + 2 * x2 * x2 - 2 * x1 - 2 * x2 + 1 - 4 * x1 * x2
    return 22 * x1 + 0.1 * x2 + np.exp(4.84 * x1 * x1 + 1.2 * x2 * x2)


def grad_function(x1, x2, delta):
    def derivative(x1, x2, delta_x1, delta_x2):
        der = ((function(x1 + delta_x1, x2 + delta_x2) - function(
            x1 - delta_x1, x2 - delta_x2)) / (
                       2 * delta))
        return der

    gradient = (
        [-1 * derivative(x1, x2, delta, 0), -1 * derivative(x1, x2, 0, delta)])
    return gradient


def gss_1(a, b, gradient, x1, x2, eps, s):
    interval = (b - a)
    a1 = a + interval * (1 - s)
    b1 = a + interval * s
    fa1 = function(x1 + a1 * gradient[0], x2 + a1 * gradient[1])
    fb1 = function(x1 + b1 * gradient[0], x2 + b1 * gradient[1])
    while abs(interval) >= eps:
        if fa1 <= fb1:  # <= - минимум, >= - максимум
            b = b1
            b1 = a1
            fb1 = fa1
            interval = interval * s
            a1 = a + interval * (1 - s)
            fa1 = function(x1 + a1 * gradient[0], x2 + a1 * gradient[1])
        else:
            a = a1
            a1 = b1
            fa1 = fb1
            interval = interval * s
            b1 = a + interval * s
            fb1 = function(x1 + b1 * gradient[0], x2 + b1 * gradient[1])
        L = (a + b) / 2
    return L


def grad_move(old_x1, old_x2, lam, gradient):
    x1 = old_x1 + lam * gradient[0]
    x2 = old_x2 + lam * gradient[1]
    old_gradient = gradient
    gradient = grad_function(x1, x2, delta)
    new_function = function(x1, x2)
    return [new_function, x1, x2, gradient, old_gradient]


def CH(grad0, grad1):
    new_grad = np.array(grad1)
    old_grad = np.array(grad0)
    khi = (np.transpose(new_grad).dot(new_grad - old_grad)) / (
        np.transpose(old_grad).dot(old_grad))
    return khi


def s_1(old_gradient, new_gradient, chi):
    sx1 = new_gradient[0] + chi * old_gradient[0]
    sx2 = new_gradient[1] + chi * old_gradient[1]
    s = [sx1, sx2]
    return s


def conj_grad(x1, x2, delta):
    points_x = [x1]
    points_y = [x2]
    func = [function(x1, x2)]
    count = 0
    new_function = function(x1, x2)
    old_function = new_function + 100
    eps_y = 0.000001
    a, b = 0, 1
    eps = (1 - a) / 100000
    x1_0, x2_0 = x1, x2
    gradient = grad_function(x1_0, x2_0, delta)
    while abs(new_function - old_function) > eps_y:
        count += 1
        lam = gss_1(a, b, gradient, x1_0, x2_0, eps, s)
        func_value, x1_1, x2_1, gradient, old_gradient = grad_move(x1_0, x2_0,
                                                                   lam,
                                                                   gradient)
        points_x.append(x1_1)
        points_y.append(x2_1)
        func.append(func_value)

        chi = CH(old_gradient, gradient)
        s1 = s_1(old_gradient, gradient, chi)
        lam = gss_1(a, b, s1, x1_1, x2_1, eps, s)
        old_function = new_function
        new_function, x1_2, x2_2, gradient = grad_move(x1_1, x2_1, lam, s1)[
                                             :-1]
        x1_0, x2_0 = x1_2, x2_2

        table(count, old_function, x1_1, x2_1, x1_0, x2_0, new_function)
        points_x.append(x1_2)
        points_y.append(x2_2)
        func.append(new_function)

    return output(x1_0, x2_0, count, eps_y), [points_x, points_y], func


if __name__ == '__main__':
    s = ((sqrt(5) - 1) / 2)
    x1 = 1
    x2 = 1
    delta = 0.000001

    info, points_coord, coord_func = conj_grad(x1, x2, delta)
    print(headers)
    print(info)

    x_axis = y_axis = np.arange(0, 2, 0.001)
    X, Y = np.meshgrid(x_axis, y_axis)
    Zs = np.array(function(np.ravel(X), np.ravel(Y)))
    Z = Zs.reshape(X.shape)
    sorted_coord_func = sorted(coord_func)
    cs = plt.contour(X, Y, Z, levels=sorted_coord_func)
    plt.clabel(cs)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.plot(points_coord[0], points_coord[1])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, zorder=2)
    ax.plot(points_coord[0], points_coord[1], coord_func, color='red',
            zorder=1)
    plt.show()
