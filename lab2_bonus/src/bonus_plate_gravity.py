import numpy as np


G = 6.674e-11


def gauss_legendre_2d(func, ax: float, bx: float, ay: float, by: float, n: int = 40) -> float:
    """二维高斯-勒让德积分：在 [ax,bx]×[ay,by] 上积分 func(x,y)。"""
    x, wx = np.polynomial.legendre.leggauss(n)
    y, wy = np.polynomial.legendre.leggauss(n)

    # 线性映射 [-1,1] -> [ax,bx], [-1,1] -> [ay,by]
    x_mapped = 0.5 * (bx - ax) * x + 0.5 * (bx + ax)
    y_mapped = 0.5 * (by - ay) * y + 0.5 * (by + ay)
    jacobian = 0.25 * (bx - ax) * (by - ay)

    integral = 0.0
    for i in range(n):
        for j in range(n):
            integral += wx[i] * wy[j] * func(x_mapped[i], y_mapped[j])

    return integral * jacobian


def plate_force_z(z: float, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40) -> float:
    """计算方板中心正上方高度 z 处的垂向引力 Fz。"""
    sigma = M_plate / L**2
    half_L = L / 2.0

    def integrand(x, y):
        r2 = x**2 + y**2 + z**2
        return z / (r2 ** 1.5)

    integral = gauss_legendre_2d(integrand, -half_L, half_L, -half_L, half_L, n=n)
    return G * m_particle * sigma * integral


def force_curve(z_values, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40):
    """对一组 z 值计算板力曲线。"""
    z_array = np.asarray(z_values, dtype=float)
    fz_values = np.empty_like(z_array, dtype=float)

    for idx, z in np.ndenumerate(z_array):
        fz_values[idx] = plate_force_z(z, L=L, M_plate=M_plate, m_particle=m_particle, n=n)

    return fz_values
