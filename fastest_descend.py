from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

headers = PrettyTable(['№', 'x1', 'x2', 'function(x1,x2)', 'dgrad'])


def table(count, x1, x2, delta_grad, function):
    Tablelist = {
        '№': count,
        'x1': round(x1, 7),
        'x2': round(x2, 7),
        'function(x1,x2)': round(function, 8),
        'dgrad': round(delta_grad, 8),
    }
    headers.add_row(Tablelist.values())


def output(x1, x2, count, eps_grad):
    return f'Число шагов = {count}\nx1 = {x1}, x2 = {x2}\nfunction(x1,x2) = {function(x1, x2)}\neps_grad = {eps_grad}'


def function(x1, x2):
    return 10 * x1 * x1 + 2 * x2 * x2 - 2 * x1 - 2 * x2 + 1 - 4 * x1 * x2
    #return 22 * x1 + 0.1 * x2 + np.exp(4.84 * x1 * x1 + 1.2 * x2 * x2)



def grad_function(x1, x2, delta):
    def derivative(x1, x2, delta_x1, delta_x2):
        der = (function(x1 + delta_x1, x2 + delta_x2) - function(x1 - delta_x1, x2 - delta_x2)) / (
                2 * delta)
        return der

    gradient = ([-1 * derivative(x1, x2, delta, 0), -1 * derivative(x1, x2, 0, delta)])
    return gradient


def delta_grad(gradient):
    d_g = sqrt(gradient[0] ** 2 + gradient[1] ** 2)
    return d_g


def ort(grad0, grad1):
    ortog = grad0[0] * grad1[0] + grad0[1] * grad1[1]
    return ortog


def gss(a, b, gradient, x1, x2, eps, s):
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


def grad_move(x10, x20, lam, gradient):
    x1 = (x10 + lam * gradient[0])
    x2 = (x20 + lam * gradient[1])
    old_gradient = gradient
    gradient = grad_function(x1, x2, delta)
    new_function = function(x1, x2)
    return [x1, x2, old_gradient, gradient, new_function]


def asc(x1, x2, delta):
    count = 0
    points_x = [x1]
    points_y = [x2]
    func = [function(x1, x2)]
    delta_gradient = 1
    eps_grad = 0.02
    a, b = 0, 1
    eps = (1 - a) / 100000
    while delta_gradient > eps_grad:
        count += 1
        gradient = grad_function(x1, x2, delta)
        lam = gss(a, b, gradient, x1, x2, eps, s)
        (x1, x2, old_gradient,
         gradient, new_function) = grad_move(x1, x2, lam, gradient)
        nf = function(x1, x2)
        delta_gradient = delta_grad(gradient)
        check = ort(old_gradient, gradient)
        eps_ort = sqrt(ort(gradient, gradient)) / 1000
        points_x.append(x1)
        points_y.append(x2)
        func.append(nf)
        if abs(check) >= abs(eps_ort):
            delta /= 10
        table(count, x1, x2, delta_gradient, function(x1, x2))
    return output(x1, x2, count, eps_grad), [points_x, points_y], func


if __name__ == '__main__':
    s = ((sqrt(5) - 1) / 2)
    x1 = 1
    x2 = 1
    delta = 0.000001

    info, points_coord, coord_func = asc(x1, x2, delta)
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
