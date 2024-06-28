from math import log, pow, e
import numpy as np
from matplotlib import pyplot as plt

# коэффициенты
r = 0.35
t_w = 2000
t_0 = 10000
p = 4
c = 3e10

a, b = 0, 1
n = 100  # число узлов
h = (b - a) / n

METHOD_AVERAGE = 1    #метод средних
METHOD_TRAPEZOID = 2  #метод трапеций

LEFT = 3
RIGHT = 4
CENTER = 5

SIMPSON = 5
TRAPEZOID = 6

#метод вычисления
methodKappa = METHOD_AVERAGE
methodProgonka = RIGHT
methodCalculateF = TRAPEZOID

# заданная таблица
t_arr = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000] 
k_1 = [8.2E-3, 2.768E-02, 6.560E-02, 1.281E-01, 2.214E-01, 3.516E-01, 5.248E-01, 7.472E-01, 1.025E+00] 
k_2 = [1.600E+00, 5.400E+00, 1.280E+01, 2.500E+01, 4.320E+01, 6.860E+01, 1.024E+02, 1.458E+02, 2.000E+02]

K_arr = k_2

def t(z):
    return (t_w - t_0) * z ** p + t_0


def u_p(z):
    return 3.084e-4 / (np.exp(4.799e4 / t(z)) - 1)

def linear_interpolation(t0, t_arr, K_arr):
    n = len(t_arr)
    j = 0

    if t0 < t_arr[0]:
        return K_arr[0]

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

def _lambda(z):    
    return c / (3 * k(z))

def _p(z):
    return c * k(z)

def _f(z):
    return c * k(z) * u_p(z)

def v_n(z, h):
    return ((z + h / 2) ** 2 - (z - h / 2) ** 2) / 2

def kappa(z1, z2):
    if methodKappa == METHOD_AVERAGE:
        return (_lambda(z1) + _lambda(z2)) / 2
    elif methodKappa == METHOD_TRAPEZOID:
        return (2 * _lambda(z1) * _lambda(z2)) / (_lambda(z1) + _lambda(z2))
    

def left_boundary_condition(z0, g0, h):

    p_half = (_p(z0) + _p(z0 + h)) / 2
    f_half = (_f(z0) + _f(z0 + h)) / 2

    k0 = kappa(z0, z0 + h) * (z0 + h / 2) / (r ** 2 * h) - p_half * (z0 + h / 2) * h / 8

    m0 = -kappa(z0, z0 + h) * (z0 + h / 2) / (r ** 2 * h) - p_half * (z0 + h / 2) * h / 8 - _p(
        z0) * z0 * h / 4

    p0 = -z0 * g0 / r - (f_half * (z0 + h / 2) + _f(z0) * z0) * h / 4

    return k0, m0, p0


def right_boundary_condition(zn, h):

    p_half = (_p(zn) + _p(zn - h)) / 2
    f_half = (_f(zn) + _f(zn - h)) / 2

    kn = kappa(zn - h, zn) * (zn - h / 2) / (r ** 2 * h) - p_half * (zn - h / 2) * h / 8

    mn = -kappa(zn - h, zn) * (zn - h / 2) / (r ** 2 * h) - 0.393 * c * zn / r - _p(
        zn) * zn * h / 4 - p_half * (zn - h / 2) * h / 8

    pn = -(_f(zn) * zn + f_half * (zn - h / 2)) * h / 4

    return kn, mn, pn


def right_progonka(a, b, h):   # Правая прогонка
    
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


def left_progonka(a, b, h):   # Левая прогонка

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


def center_progonka(a, b, h, n_eq):   #Встречная прогонка
    # Прямой ход
    k0, m0, p0 = left_boundary_condition(a, 0, h)
    kn, mn, pn = right_boundary_condition(b, h)

    # правая часть прогонки
    z = h
    n = 1

    ksi_r = [0, -k0 / m0]
    eta_r = [0, p0 / m0]

    while z < n_eq * h + h / 2:
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

    u[n_eq] = (ksi_r[-1] * eta_l[0] + eta_r[-1]) / (1 - ksi_r[-1] * ksi_l[0])

    for i in range(n_eq - 1, -1, -1):
        u[i] = ksi_r[i + 1] * u[i + 1] + eta_r[i + 1]

    for i in range(n_eq + 1, n + cnt):
        _i = i - n_eq
        u[i] = ksi_l[_i - 1] * u[i - 1] + eta_l[_i - 1]

    return u

def progonka():
    if methodProgonka == LEFT:
        return left_progonka(a, b, h)
    elif methodProgonka == RIGHT:
        return right_progonka(a, b, h)
    else:
        return center_progonka(a, b, h, n // 2)

def getF1(z, u):
    #аппроксимацией производной центральным аналогом
    # (2-й порядок точности)

    f_res = [0]

    for i in range(1, len(u) - 1):
        f_res.append(-(_lambda(z[i]) / r) * (u[i + 1] - u[i - 1]) / (2 * h))

    f_res.append(-(_lambda(z[len(u) - 1]) / r) * (3 * u[-1] - 4 * u[-2] + u[-3]) / (2 * h)) 
    # Односторонняя разностная аппроксимация

    return f_res


def getF2(z, u): #трапеций (линейный)
    
    _f = [0]
    f_res = [0]
    print(len(z), len(u))
    for i in range(1, len(z)):
        _f.append(k(z[i]) * (u_p(z[i]) - u[i]) * z[i])
        f_res.append((c * r / z[i]) * h * ((_f[0] + _f[i]) / 2 + sum(_f[1:-1]))) #?

    return f_res


def getF3(z, u):  #симпсона (парабольный)
    
    _f = [0]
    f_res = [0]

    for i in range(1, len(z)):
        _f.append(k(z[i]) * (u_p(z[i]) - u[i]) * z[i])

        _sum = 0

        for _k in range(1, len(z[:i]), 2):
            _sum += (_f[_k - 1] + 4 * _f[_k] + _f[_k + 1])

        f_res.append((c * r / z[i]) * (h / 3) * _sum)

    return f_res

def getF(z, u):
    if methodCalculateF == TRAPEZOID:
        return getF2(z, u)
    elif methodCalculateF == SIMPSON:
        return getF3(z, u)



def main():
    u_res = progonka()
    z_res = np.arange(a, b + h / 2, h)

    f_res = getF1(z_res, u_res)
    
    f_res2 = getF(z_res, u_res)

    up_res = [0] * len(z_res)
    div_f = [0] * len(z_res)

    for i in range(len(z_res)):
        up_res[i] = u_p(z_res[i])
        div_f[i] = div_flux(z_res[i], u_res[i])

    plt.figure(figsize=(9, 6))
    plt.subplot(2, 2, 1)
    plt.plot(z_res, u_res, 'r', label='u(z)')
    plt.plot(z_res, up_res, 'b', label='u_p')
    plt.legend()
    plt.grid()
    #
    plt.subplot(2, 2, 2)
    plt.plot(z_res, f_res, 'g', label='F(z)')
    plt.legend()
    plt.grid()
    #
    plt.subplot(2, 2, 3)
    plt.plot(z_res, f_res2, 'g', label='F(z) integral')
    plt.legend()
    plt.grid()
    #
    plt.subplot(2, 2, 4)
    plt.plot(z_res, div_f, 'y', label='divF(z)')
    plt.legend()
    plt.grid()
    #
    plt.show()

if __name__ == '__main__':
    main()
