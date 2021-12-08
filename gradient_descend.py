from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

headers = PrettyTable(['№', 'x1', 'x2', 'function(x1,x2)', 'df'])


def table(count, old_function, x1, x2, new_function):
    table_list = {
        '№': count,
        'x1': round(x1, 7),
        'x2': round(x2, 7),
        'function(x1,x2)': round(new_function, 8),
        'df': abs(round(new_function - old_function, 8)),
    }
    headers.add_row(table_list.values())


def function(x, y):
    return 22 * x + 0.1 * y + np.exp(4.84 * x * x + 1.2 * y * y)


def output(x1, x2, count, lam, eps_y):
    return (f'Число шагов = {count}\nx1 = {x1}, x2 = {x2}'
            f'\nfunction(x1,x2) = {function(x1, x2)}\n'
            f'lambda = {lam}, eps_y = {eps_y} ')


def grad_function(X, delta):
    def derivative(x1, x2, delta_x1, delta_x2):
        der = (function(x1 + delta_x1, x2 + delta_x2)
               - function(x1 - delta_x1, x2 - delta_x2)) / (2 * delta)
        return der

    gradient = np.array([-1 * derivative(*X, delta, 0),
                         -1 * derivative(*X, 0, delta)])
    return gradient


def grad_move(X, delta, lamb):
    count = 0
    points_x = [X[0]]
    points_y = [X[1]]
    func = [function(*X)]
    eps_y = 5e-14
    gradient = grad_function(X, delta)
    delta_grad = 1
    while delta_grad > 0.0001:
        count += 1
        OLD_X = X
        X = OLD_X + lamb * gradient
        gradient = grad_function(X, delta)
        delta_grad = sqrt(np.dot(gradient, gradient))
        points_x.append(X[0])
        points_y.append(X[1])
        func.append(function(*X))
    table(count, 0, *X, 0)
    return output(*X, count, lamb, eps_y), [points_x, points_y], func


if __name__ == '__main__':
    x1 = -1
    x2 = -1
    X = np.array([x1, x2])
    delta_x1 = delta_x2 = 0.0001
    lambd = 0.0001
    info, points_coord, coord_func = grad_move(X, delta_x1, lambd)
    print(headers)
    print(info)

    x_axis = y_axis = np.arange(-1, 1, 0.001)
    X, Y = np.meshgrid(x_axis, y_axis)
    Zs = np.array(function(np.ravel(X), np.ravel(Y)))
    Z = Zs.reshape(X.shape)
    sorted_coord_func = sorted(coord_func)
    step = len(coord_func) // 50
    cs = plt.contour(X, Y, Z, levels=sorted_coord_func[::step])
    plt.clabel(cs)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.plot(points_coord[0], points_coord[1])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, zorder=2)
    ax.plot(points_coord[0], points_coord[1], coord_func, color='red', zorder=1)
    plt.show()
