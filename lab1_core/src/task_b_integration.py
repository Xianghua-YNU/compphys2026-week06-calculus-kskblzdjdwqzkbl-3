import math


def debye_integrand(x: float) -> float:
    if abs(x) < 1e-12:
        return 0.0
    ex = math.exp(x)
    return (x**4) * ex / ((ex - 1.0) ** 2)


def trapezoid_composite(f, a: float, b: float, n: int) -> float:

    # 实现复合梯形积分
    h = (b - a) / n
    integral = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        integral += f(a + i * h)
    return integral * h


def simpson_composite(f, a: float, b: float, n: int) -> float:
    # 实现复合 Simpson 积分，并检查 n 为偶数
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")
    h = (b - a) / n
    integral = f(a) + f(b)
    for i in range(1, n, 2):
        integral += 4 * f(a + i * h)
    for i in range(2, n, 2):
        integral += 2 * f(a + i * h)
    return integral * h / 3


def debye_integral(T: float, theta_d: float = 428.0, method: str = "simpson", n: int = 200) -> float:
    # 计算 Debye 积分 I(theta_d/T)
    x = theta_d / T
    if method == "simpson":
        return simpson_composite(debye_integrand, 0, x, n)
    elif method == "trapezoid":
        return trapezoid_composite(debye_integrand, 0, x, n)
    else:
        raise ValueError("method must be 'simpson' or 'trapezoid'")

    # 复合梯形积分
    if n <= 0:
        raise ValueError("n must be positive")

    h = (b - a) / n
    s = 0.5 * f(a) + 0.5 * f(b)

    for i in range(1, n):
        x = a + i * h
        s += f(x)

    return h * s


def simpson_composite(f, a: float, b: float, n: int) -> float:
    # 复合 Simpson 积分，要求 n 为偶数
    if n <= 0:
        raise ValueError("n must be positive")
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson composite rule")

    h = (b - a) / n
    s = f(a) + f(b)

    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            s += 2.0 * f(x)
        else:
            s += 4.0 * f(x)

    return (h / 3.0) * s


def debye_integral(T: float, theta_d: float = 428.0, method: str = "simpson", n: int = 200) -> float:
    # 计算 Debye 积分 I(theta_d / T)
    if T <= 0:
        raise ValueError("T must be positive")

    y = theta_d / T

    if method == "trapezoid":
        return trapezoid_composite(debye_integrand, 0.0, y, n)
    elif method == "simpson":
        return simpson_composite(debye_integrand, 0.0, y, n)
    else:
        raise ValueError("method must be 'trapezoid' or 'simpson'")


if __name__ == "__main__":
    T = 300.0
    theta_d = 428.0
    n = 200

    trap = debye_integral(T, theta_d, method="trapezoid", n=n)
    simp = debye_integral(T, theta_d, method="simpson", n=n)

    print(f"T = {T}, theta_d = {theta_d}, n = {n}")
    print(f"Trapezoid: {trap}")
    print(f"Simpson:   {simp}")
    print(f"Difference = {abs(trap - simp)}")

