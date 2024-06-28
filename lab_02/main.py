from math import exp, log, pow, e
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

EPS = 1e-4
c = 3e10
m = 0.786
R = 0.35
Tw = 2000
T0 = 1e4
p_range = [4, 15 + 1]
p_range = [4, 5]

table1 = pd.read_csv("table1.csv")

t_arr = list(table1["T"])
K1_arr = list(table1["K2"])

def T(z, p):
    return (Tw - T0) * (z ** p) + T0

def up(z, p):
    return 3.084e-4 / (exp(4.799e4 / T(z, p)) - 1)


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

def k(z, p):
    return float(linear_interpolation(T(z, p), t_arr, K1_arr))

def psi(uF_tuple):
    u, F = uF_tuple
    return F - m * c * u / 2


class RungeKutta4Solver:
    def __init__(self, f_func, phi_func):
        self.f = f_func
        self.phi = phi_func

    def next_xyz(self, xn, yn, zn, h):
        k1 = h * self.f(xn, yn, zn)
        q1 = h * self.phi(xn, yn, zn)
        q2 = h * self.phi(xn + h / 2, yn + k1 / 2, zn + q1 / 2)
        k2 = h * self.f(xn + h / 2, yn + k1 / 2, zn + q2 / 2)
        k3 = h * self.f(xn + h / 2, yn + k2 / 2, zn + q2 / 2)
        q3 = h * self.phi(xn + h / 2, yn + k2 / 2, zn + q2 / 2)
        k4 = h * self.f(xn + h, yn + k3, zn + q3)
        q4 = h * self.phi(xn + h, yn + k3, zn + q3)

        x_n_plus_1 = xn + h
        y_n_plus_1 = yn + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        z_n_plus_1 = zn + (q1 + 2 * q2 + 2 * q3 + q4) / 6

        return x_n_plus_1, y_n_plus_1, z_n_plus_1

    def solve(self, x_start, y_start, z_start, x_max, h, show_result=False):
        solution_steps = []
        x, y, z = x_start, y_start, z_start

        while x < x_max + EPS:
            solution_steps.append([x, y, z])
            x, y, z = self.next_xyz(x, y, z, h)
            if (y < -1e-6):
                y = up(x, p_range[0])

        if show_result:
            process_results(solution_steps)

        return tuple((y, z))


class HalfDivisionSolver:
    def __init__(self, f_func):
        self.f = f_func

    def solve(self, x_min, x_max):
        x1 = x_min
        x2 = x_max
        x = (x1 + x2) / 2

        while abs((x1 - x2) / x) > EPS:
            cur_y = self.f(x)

            if self.f(x1) * cur_y > 0:
                x1 = x
            else:
                x2 = x

            x = (x1 + x2) / 2

        return x


P_SHOW = 0


def process_results(solution_steps):
    global P_SHOW
    x, y, z = [step[0] for step in solution_steps], \
              [step[1] for step in solution_steps], \
              [step[2] for step in solution_steps]
    table = pd.DataFrame(index=x)
    table['x (z)'] = x
    table = table.set_index('x (z)')
    table['y (u)'] = y
    table['z (F)'] = z

    print(y[0], z[0])
    show_each = 1

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    # print(f'P: {P_SHOW}')
    # print(table.iloc[::show_each, :])
    # print('\n\n\n')

    plt.subplot(3, 1, 1)
    plt.plot(x, y, '.', label='u(z)')
    plt.legend()

    ups = list(map(lambda x0: up(x0, P_SHOW), x))
    plt.subplot(3, 1, 2)
    plt.plot(x, ups, '-', label='up(z)')
    
    plt.legend()
    plt.subplot(3, 1, 3)
    #plt.ylim([0, 9000])
    plt.plot(x, z, '--', label='F(z)')
    plt.legend()
    plt.xlabel('z')
    # plt.grid()
    plt.savefig(f'p{P_SHOW}.png')
    plt.show()


def main():
    h = 1e-2

    z_min = 0
    z_max = 1

    F_min = 0

    ksi_min = 0.01
    ksi_max = 1

    for p in range(*p_range):
        rk_f_func = lambda x, u, v: -3 * R * v * k(x, p) / c
        rk_phi_func = lambda x, u, v: (
            R * c * k(x, p) * (up(x, p) - u) - (v / x) if x != 0  # zero division
            else
            R * c * k(x, p) * (up(x, p) - u) / 2)
        u_min_func = lambda ksi: ksi * up(0, p)

        hd_f_func = lambda ksi: \
            psi(RungeKutta4Solver(rk_f_func, rk_phi_func).solve(z_min, u_min_func(ksi), F_min, z_max, h))

        ksi = HalfDivisionSolver(hd_f_func).solve(ksi_min, ksi_max)

        print(RungeKutta4Solver(rk_f_func, rk_phi_func).solve(z_min, u_min_func(ksi), F_min, z_max, h))

        global P_SHOW
        P_SHOW = p
        RungeKutta4Solver(rk_f_func, rk_phi_func).solve(z_min, u_min_func(ksi), F_min, z_max, h, show_result=True)
        print(f'p: {p}, solution_ksi: {ksi}')


if __name__ == '__main__':
    main()