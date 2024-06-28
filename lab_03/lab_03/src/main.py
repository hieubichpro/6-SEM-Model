from math import exp, log, pow, e
import numpy as np
import panda as pd
from matplotlib import pyplot as plt

# коэффициенты
r = 0.35
t_w = 2000
t_0 = 1e4
p = 4
c = 3e10
EPS = 1e-4

METHOD_AVERAGE = 1    #метод средних
METHOD_TRAPEZOID = 2  #метод трапеций

methodKappa = 1

# заданная таблица
table1 = pd.read_csv("table1.csv")

t_arr = list(table1["T"])
K_arr = list(table1["K2"])

def t(z):
    return (t_w - t_0) * z ** p + t_0


def u_p(z):
    return 3.084e-4 / (np.exp(4.799e4 / t(z)) - 1)

def linear_interpolation(t0, t_arr, K_arr):
    n = len(t_arr)
    j = 0

    if t0 < t_arr[0]:
        t = t_arr[0]
        K = K_arr[0]
        return K

    while True:
        if t_arr[j] > t0 or j == n - 2:
            break
        j += 1
    j -= 1

    if j < n - 1:
        dt = log(t_arr[j + 1]) - log(t_arr[j])
        dk = log(K_arr[j + 1]) - log(K_arr[j])

        K = log(K_arr[j]) + (log(t0) - log(t_arr[j])) * dk / dt
    else:
        dt = log(t_arr[n - 1]) - log(t_arr[n - 2])
        dk = log(K_arr[n - 1]) - log(K_arr[n - 2])

        K = log(K_arr[n - 2]) + (log(t0) - log(t_arr[n -2])) * dk / dt

    return pow(e, K)

def k(z):
    return float(linear_interpolation(t(z), t_arr, K_arr))

def div_flux(z, u): #ok
    return c * k(z) * (u_p(z) - u)

def _lambda(z):    # без скорости света
    return 1 / (3 * k(z))

def _p(z):
    return k(z)


def _f(z):
    return k(z) * u_p(z)

def v_n(z, h):
    return ((z + h / 2) ** 2 - (z - h / 2) ** 2) / 2

def kappa(z1, z2):
    if methodKappa == METHOD_AVERAGE:
        return (_lambda(z1) + _lambda(z2)) / 2
    elif methodKappa == METHOD_TRAPEZOID:
        return (2 * _lambda(z1) * _lambda(z2)) / (_lambda(z1) + _lambda(z2))
    

def left_boundary_condition(z0, g0, h):
    """
    Левое краевое условие метода правой прогонки
    """

    p_half = (_p(z0) + _p(z0 + h)) / 2
    f_half = (_f(z0) + _f(z0 + h)) / 2

    k0 = kappa(z0, z0 + h) * (z0 + h / 2) / (r ** 2 * h) - p_half * (z0 + h / 2) * h / 8

    m0 = -kappa(z0, z0 + h) * (z0 + h / 2) / (r ** 2 * h) - p_half * (z0 + h / 2) * h / 8 - _p(
        z0) * z0 * h / 4

    p0 = -z0 * g0 / r - (f_half * (z0 + h / 2) + _f(z0) * z0) * h / 4

    return k0, m0, p0


def right_boundary_condition(zn, h):
    """
    Правое краевое условие метода правой прогонки
    """
    p_half = (_p(zn) + _p(zn - h)) / 2
    f_half = (_f(zn) + _f(zn - h)) / 2

    kn = kappa(zn - h, zn) * (zn - h / 2) / (r ** 2 * h) - p_half * (zn - h / 2) * h / 8  # правильно

    mn = -kappa(zn - h, zn) * (zn - h / 2) / (r ** 2 * h) - 0.393 * 1 * zn / r - _p(
        zn) * zn * h / 4 - p_half * (zn - h / 2) * h / 8

    pn = -(_f(zn) * zn + f_half * (zn - h / 2)) * h / 4

    return kn, mn, pn


def right_progonka(a, b, h):
    """
    Реализация правой прогонки
    """
    # Прямой ход
    k0, m0, p0 = left_boundary_condition(a, 0, h)
    kn, mn, pn = right_boundary_condition(b, h)

    ksi = [0, -k0 / m0]
    eta = [0, p0 / m0]

    z = h
    n = 1

    while z < b + h / 2:
        a_n = (z - h / 2) * (kappa(z - h, z)) / (r ** 2 * h)
        c_n = ((z + h / 2) * kappa(z, z + h)) / (r ** 2 * h)
        b_n = a_n + c_n + _p(z) * v_n(z, h)
        d_n = _f(z) * v_n(z, h)

        ksi.append(c_n / (b_n - a_n * ksi[n]))
        eta.append((a_n * eta[n] + d_n) / (b_n - a_n * ksi[n]))

        n += 1
        z += h

    # Обратный ход
    u = [0] * n

    u[n - 1] = (pn - kn * eta[n - 1]) / (kn * ksi[n - 1] + mn)

    for i in range(n - 2, -1, -1):
        u[i] = ksi[i + 1] * u[i + 1] + eta[i + 1]

    return u


def left_progonka(a, b, h):
    """
    Реализация левой прогонки
    """

    # Прямой ход
    k0, m0, p0 = left_boundary_condition(a, 0, h)
    kn, mn, pn = right_boundary_condition(b, h)

    ksi = [-kn / mn, 0]
    eta = [pn / mn, 0]

    z = b - h
    n = -2
    cnt = 1

    while z > a - h / 2:
        a_n = (z - h / 2) * (kappa(z - h, z)) / (r ** 2 * h)
        c_n = ((z + h / 2) * kappa(z, z + h)) / (r ** 2 * h)
        b_n = a_n + c_n + _p(z) * v_n(z, h)
        d_n = _f(z) * v_n(z, h)

        ksi.insert(0, a_n / (b_n - c_n * ksi[n]))
        eta.insert(0, (c_n * eta[n] + d_n) / (b_n - c_n * ksi[n]))

        n -= 1
        z -= h
        cnt += 1

    # Обратный ход
    u = [0] * cnt

    u[0] = (p0 - k0 * eta[0]) / (m0 + k0 * ksi[0])

    for i in range(1, cnt):
        u[i] = ksi[i - 1] * u[i - 1] + eta[i - 1]

    return u


def center_progonka(a, b, h, n_eq):
    """
    Реализация встречной прогонки
    """
    # Прямой ход
    k0, m0, p0 = left_boundary_condition(a, 0, h)
    kn, mn, pn = right_boundary_condition(b, h)

    # правая часть прогонки
    z = h
    n = 1

    ksi_r = [0, -k0 / m0]
    eta_r = [0, p0 / m0]

    while z < n_eq * h - h / 2:
        a_n = (z - h / 2) * (kappa(z - h, z)) / (r ** 2 * h)
        c_n = ((z + h / 2) * kappa(z, z + h)) / (r ** 2 * h)
        b_n = a_n + c_n + _p(z) * v_n(z, h)
        d_n = _f(z) * v_n(z, h)

        ksi_r.append(c_n / (b_n - a_n * ksi_r[n]))
        eta_r.append((a_n * eta_r[n] + d_n) / (b_n - a_n * ksi_r[n]))

        n += 1
        z += h

    # левая часть прогонки
    ksi_l = [-kn / mn, 0]
    eta_l = [pn / mn, 0]

    z = b - h
    n1 = -2
    cnt = 1

    while z > n_eq * h:
        print(f"[+] in while")
        a_n = (z - h / 2) * (kappa(z - h, z)) / (r ** 2 * h)
        c_n = ((z + h / 2) * kappa(z, z + h)) / (r ** 2 * h)
        b_n = a_n + c_n + _p(z) * v_n(z, h)
        d_n = _f(z) * v_n(z, h)

        ksi_l.insert(0, a_n / (b_n - c_n * ksi_l[n1]))
        eta_l.insert(0, (c_n * eta_l[n1] + d_n) / (b_n - c_n * ksi_l[n1]))

        n1 -= 1
        z -= h
        cnt += 1

    # # Обратный ход
    u = [0] * (n + cnt)

    # вычисляем up (решая систему из двух уравнений) -- сопряжение решений
    u[n_eq] = (ksi_r[-1] * eta_l[0] + eta_r[-1]) / (1 - ksi_r[-1] * ksi_l[0])

    # print(f"u[n_eq] = u[{n_eq}] = {u[n_eq]: <.7e}, len(u) = {len(u)}")
    # # print(f"U в точке p = {n_eq} равно {u[n_eq]: <.7e}")

    for i in range(n_eq - 1, -1, -1):
        u[i] = ksi_r[i + 1] * u[i + 1] + eta_r[i + 1]

    for i in range(n_eq + 1, n + cnt):
        print(f"[+] in for")
        _i = i - n_eq
        u[i] = ksi_l[_i - 1] * u[i - 1] + eta_l[_i - 1]

    return u


def flux1(u, z, h):
    """
    Вычисление F(z) аппроксимацией производной центральным аналогом
    (2-й порядок точности)
    """
    f_res = [0]

    for i in range(1, len(u) - 1):
        curr_f = -(_lambda(z[i]) / r) * (u[i + 1] - u[i - 1]) / (2 * h)
        f_res.append(curr_f)

    f_res.append(-(_lambda(z[len(u) - 1]) / r) * (3 * u[-1] - 4 * u[-2] + u[-3]) / (2 * h)) 
    # Односторонняя разностная аппроксимация

    return f_res


def flux2(z, u, h): #?
    """
    Метод трапеций для вычисления интеграла при получении F(z)
    """
    _f = [0]
    f_res = [0]

    for i in range(1, len(z)):
        _f.append(k(z[i]) * (u_p(z[i]) - u[i]) * z[i])
        f_res.append((c * r / z[i]) * h * ((_f[0] + _f[i]) / 2 + sum(_f[1:-1]))) #?

    return f_res


def flux3(z, u, h):
    """
    Метод Симпсона для вычисления интеграла при получении F(z)
    """
    _f = [0]
    f_res = [0]

    for i in range(1, len(z)):
        _f.append(k(z[i]) * (u_p(z[i]) - u[i]) * z[i])

        _sum = 0

        for _k in range(1, len(z[:i]), 2):
            _sum += (_f[_k - 1] + 4 * _f[_k] + _f[_k + 1])

        f_res.append((c * r / z[i]) * (h / 3) * _sum)

    return f_res


def get_research():
    """
    Исследование правой прогонки по полной
    """
    table_size = 85
    a, b = 0, 1

    n_list = [100, 70, 50, 30, 20, 10]

    file = open("../data/research.txt", "w", encoding="utf-8")

    file.write("-" * table_size + "\n")
    file.write(f' {"n": ^7} | {"u(0)": ^22} | {"u(1)": ^22} | {"f(1)": ^22} |\n')
    file.write("-" * table_size + "\n")

    for n in n_list:
        h = (b - a) / n

        u_res = right_progonka(a, b, h)
        # u_res = left_progonka(a, b, h)
        z_res = np.arange(a, b + h / 2, h)
        f_res = flux3(z_res, u_res, h)

        file.write(f"{n: 8} | {u_res[0]: ^22.6e} | {u_res[-1]: ^22.6e} | {f_res[-1]: ^22.6e} |\n")

    file.write("-" * table_size)

    file.close()


def write_result_to_file(filepath, z_res, u_res, f_res, f_res2):
    """
    Запись результатов в файл
    """
    file = open(filepath, "w", encoding="utf-8")

    file.write(f"Число узлов n = {len(z_res)}\n")
    file.write("-" * 86 + "\n")
    file.write(f'| {"x": ^7} | {"u(z)": ^22} | {"f(z)": ^22} | {"f(z) integral": ^22} |\n')
    file.write("-" * 86 + "\n")

    # for i in range(len(z_res)):
    #     file.write(f"| {z_res[i]: ^7.5f} | {u_res[i]: ^22.6e} | {f_res[i]: ^22.6e} | {f_res2[i]: ^22.6e} |\n")

    for i in [0, len(u_res) - 1]:
        file.write(f"| {z_res[i]: ^7.5f} | {u_res[i]: ^22.6e} | {f_res[i]: ^22.6e} | {f_res2[i]: ^22.6e} |\n")

    file.write("-" * 86)

    file.close()


def main():
    get_research()
    a, b = 0, 1
    n = 100  # число узлов
    h = (b - a) / n

    u_res = right_progonka(a, b, h)
    # u_res = left_progonka(a, b, h)
    # u_res = center_progonka(a, b, h, n)
    z_res = np.arange(a, b + h / 2, h)
    #
    f_res = flux1(u_res, z_res, h)
    # # f_res2 = flux2(z_res, u_res, h)
    # f_res2 = flux3(z_res, u_res, h)
    # f_res2 = [0] * len(z_res)
    # for i in range(1, len(z_res)):
    # f_res2[i] = flux2(z_res[i], h, u_res[i - 1], u_res[i])
    #
    up_res = [0] * len(z_res)
    div_f = [0] * len(z_res)
    #
    for i in range(len(z_res)):
        up_res[i] = u_p(z_res[i])
        div_f[i] = div_flux(z_res[i], u_res[i])

    # write_result_to_file("../data/right_progonka.txt", z_res, u_res, f_res, f_res2)
    # write_result_to_file("../data/left_progonka.txt", z_res, u_res, f_res, f_res2)
    # write_result_to_file("../data/center_progonka.txt", z_res, u_res, f_res, f_res2)

    plt.figure(figsize=(9, 6))
    plt.subplot(2, 2, 1)
    plt.plot(z_res, u_res, 'r', label='u(z)')
    plt.plot(z_res, up_res, 'b', label='u_p')
    plt.legend()
    plt.grid()
    #
    # plt.subplot(2, 2, 2)
    # plt.plot(z_res, f_res, 'g', label='F(z)')
    # plt.legend()
    # plt.grid()
    #
    # plt.subplot(2, 2, 3)
    # plt.plot(z_res, f_res2, 'g', label='F(z) integral')
    # plt.legend()
    # plt.grid()
    #
    # plt.subplot(2, 2, 4)
    # plt.plot(z_res, div_f, 'y', label='divF(z)')
    # plt.legend()
    # plt.grid()
    #
    # plt.show()

    # cmp_res_by_input_data(a, b, h)


if __name__ == '__main__':
    main()
