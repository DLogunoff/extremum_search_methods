from math import exp, log

from prettytable import PrettyTable

import numpy as np
import matplotlib.pyplot as plt


def Output(a, b, n, n_exp, eps, h):
    return (f'Ожидаемое количество шагов: {n_exp}\n'
            f'Количество шагов: {n}\n'
            f'Конечный интервал неопределенности: {a}-{b}\n'
            f'Экстремум функции лежит в точке: {extremum(a, b)}\n'
            f'Экстремальное значение функции: {calculate_function(extremum(a, b))}\n'
            f'b - a = {b - a}, h = {h}, eps = {eps}\n')


def tableBis(n, a, a1, b, b1, fa1, fb1):
    listBis = {
        '№': n,
        'a': round(a, 4),
        'a1': round(a1, 4),
        'b': round(b, 4),
        'b1': round(b1, 4),
        'fa1': round(fa1, 5),
        'fb1': round(fb1, 5),
        'b-a': round(b - a, 5),
    }
    headersBis.add_row(listBis.values())


def calculate_function(x):
    return (0.8 * x) + exp(abs(x - 1.8))


def extremum(a, b):
    return (a + b) / 2


def bisection(a, b):
    extremums = []
    functions = []
    n = 0
    eps = (b - a) / 1000
    h = 0.1 * eps
    n_expected = log((b - a) / (eps - 2 * h), 2)
    while abs(b - a) >= eps:
        n += 1
        a1 = ((a + b) / 2) - h
        b1 = ((a + b) / 2) + h
        fa1 = calculate_function(a1)
        fb1 = calculate_function(b1)
        if fa1 >= fb1:  # >= ищем минимум, <= ищем максимум
            a = a1
        else:
            b = b1
        tableBis(n, a, a1, b, b1, fa1, fb1)
        extremums.append(extremum(a, b))
        functions.append(calculate_function(extremums[-1]))
    print(headersBis)
    return Output(a, b, n, n_expected, eps, h), extremums, functions


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
    headersBis = PrettyTable(['№', 'a', 'a1', 'b', 'b1', 'fa1', 'fb1', 'b-a'])
    start = 0
    end = 3
    info, extremum_coord, extremum_func = bisection(start, end)
    print(f'Метод деления пополая\n{info}')
    x_axis = np.arange(start, end, 0.0001)
    y_axis = list(map(calculate_function, x_axis))
    plt.plot(x_axis, y_axis, 'blue')
    scatter = get_points_size(extremum_coord, 100, 10)
    plt.scatter(extremum_coord, extremum_func, s=scatter, color='red')
    plt.show()

