import math
import numpy as np

# Euler
def du_dx(x, u, v):
    return v

def dv_dx(x, u, v):
    return -0.1 * v**2 - (1 + 0.1 * x) * u

def euler_method(func1, func2, x0, u0, v0, h, num_steps):
    x_values = [x0]
    u_values = [u0]
    v_values = [v0]

    for _ in range(num_steps):
        x = x_values[-1]
        u = u_values[-1]
        v = v_values[-1]

        u_next = u + h * func1(x, u, v)
        v_next = v + h * func2(x, u, v)
        x_next = x + h

        x_values.append(x_next)
        u_values.append(u_next)
        v_values.append(v_next)

    return x_values, u_values, v_values


#taylor
def taylor(x):
    return 1 + 2 * x - 0.7 * x * x - 77 / 300 * x * x * x + 153 / 5125 * x * x * x * x 

#picard
def picard(x):
    return 1 + 2 * x - 0.7 * x * x + 0.23 / 3 * x * x * x - 11 / 750 * x * x * x * x

def main():
    x_start = 0  # начальное значение
    x_end = 1  # конечное значение
    h = 1e-5  # приближение

    n = math.ceil(abs(x_end - x_start) / h) + 1  # число итераций ~ 4000
    # output_step = int(n / 200)  # выводим только 200 значений в таблице
    output_step = 10

    arr_x, u_euler, v_euler = euler_method(du_dx, dv_dx, x_start, 1, 2, h, n)

    print(
        "------------------------------------------------------------------------------------------")
    print(
        "|    x    |  Метод Эйлера | Метод Пикара |  Разложения в степенной ряд |")
    print(
        "------------------------------------------------------------------------------------------")

    for i in range(0, n, output_step):
        print("|{:^9.3f}|{:^14f}|{:^15.3f}|{:^15.3f}        |".format(x_start, u_euler[i], picard(x_start), taylor(x_start)))
        # print("|{:^9.3f}|{:^14f}|{:^15.3f}|{:^15.3f}        |".format(x_start, u_euler[i], u_picard[i], taylor(x_start)))
        x_start += h * output_step
    print()

        
if __name__ == "__main__":
    main()
