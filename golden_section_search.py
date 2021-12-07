from math import exp, log, sqrt

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def Output(a, b, n, n_exp, eps):
    return (f'Ожидаемое количество шагов: {n_exp}\n'
            f'Количество шагов: {n}\n'
            f'Конечный интервал неопределенности: {a}-{b}\n'
            f'Экстремум функции лежит в точке: {extremum(a, b)}\n'
            f'Экстремальное значение функции: {calculate_function(extremum(a, b))}\n'
            f'b - a = {b - a}, eps = {eps}\n')


def tableGSS(n, a, a1, b1, b, fa1, fb1, interval):
    listGSS = {
        '№': n,
        'a': round(a, 4),
        'a1': round(a1, 4),
        'b1': round(b1, 4),
        'b': round(b, 4),
        'fa1': round(fa1, 5),
        'fb1': round(fb1, 5),
        'b-a': round(interval, 5),
    }
    headersGSS.add_row(listGSS.values())


def calculate_function(x):
    return (0.8 * x) + exp(abs(x - 1.8))


def extremum(a, b):
    return (a + b) / 2


def gss(a, b):
    extremums = []
    functions = []
    interval = (b - a)
    eps = interval / 1000
    s = (sqrt(5) - 1) / 2
    n_expected = log((b - a) / eps, 1.618)
    a1 = a + interval * (1 - s)
    b1 = a + interval * s
    fb1 = calculate_function(b1)
    fa1 = calculate_function(a1)
    n = 0
    tableGSS(n, a, a1, b1, b, fa1, fb1, interval)
    while abs(b - a) >= eps:
        n += 1
        if fa1 <= fb1:  # <= - минимум, >= - максимум
            b = b1
            b1 = a1
            fb1 = fa1
            interval = interval * s
            a1 = a + interval * (1 - s)
            fa1 = calculate_function(a1)
        else:
            a = a1
            a1 = b1
            fa1 = fb1
            interval = interval * s
            b1 = a + interval * s
            fb1 = calculate_function(b1)
        tableGSS(n, a, a1, b1, b, fa1, fb1, interval)
        extremums.append(extremum(a, b))
        functions.append(calculate_function(extremums[-1]))
    return Output(a, b, n, n_expected, eps), extremums, functions


def get_points_size(points, max_size, min_size):
    result = []
    amount = len(points)
    decrement_of_size = (max_size - min_size) / (amount//2)
    i = 0
    while i != amount:
        if i <= amount // 2:
            result.append(max_size - decrement_of_size * i)
        else:
            result.append(min_size)
        i += 1
    return result


if __name__ == '__main__':
    headersGSS = PrettyTable(['№', 'a', 'a1', 'b1', 'b', 'fa1', 'fb1', 'b-a'])
    start = 0
    end = 3
    info, extremum_coord, extremum_func = gss(start, end)
    print(headersGSS)
    print(f'МЕТОД ЗОЛОТОГО СЕЧЕНИЯ\n{info}')
    x_axis = np.arange(start, end, 0.0001)
    y_axis = list(map(calculate_function, x_axis))
    plt.plot(x_axis, y_axis, 'blue')
    scatter = get_points_size(extremum_coord, 100, 10)
    plt.scatter(extremum_coord, extremum_func, s=scatter, color='red')
    plt.show()
