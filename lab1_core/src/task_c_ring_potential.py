import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def ring_potential_point(x, y, z, a=1.0, q=1.0, n_phi=720):
    """
    计算均匀带电圆环在空间点(x, y, z)处的电势。

    参数:
    x, y, z: 空间点的坐标
    a: 圆环半径，默认1.0
    q: 电荷参数，默认1.0
    n_phi: 积分点数（兼容性参数，不使用）

    返回:
    电势值
    """
    def integrand(phi):
        return 1 / np.sqrt((x - a * np.cos(phi))**2 + (y - a * np.sin(phi))**2 + z**2 + 1e-16)

    integral, _ = quad(integrand, 0, 2 * np.pi)
    return (q / (2 * np.pi)) * integral

def ring_potential_grid(y_grid, z_grid, x0=0.0, a=1.0, q=1.0, n_phi=720):
    """
    计算x=x0平面上的yz网格点的电势。

    参数:
    y_grid, z_grid: 一维或二维numpy数组，网格坐标
    x0: 固定x坐标，默认0.0
    a: 圆环半径，默认1.0
    q: 电荷参数，默认1.0
    n_phi: 积分点数（兼容性参数，不使用）

    返回:
    二维numpy数组，电势值
    """
    if y_grid.ndim == 1 and z_grid.ndim == 1:
        Y, Z = np.meshgrid(y_grid, z_grid)
    else:
        Y, Z = y_grid, z_grid
    # 使用向量化方式调用ring_potential_point
    vectorized_potential = np.vectorize(lambda y, z: ring_potential_point(x0, y, z, a, q, n_phi))
    return vectorized_potential(Y, Z)

def axis_potential_analytic(z, a=1.0, q=1.0):
    """
    计算z轴上圆环电势的解析解。

    参数:
    z: z坐标
    a: 圆环半径，默认1.0
    q: 电荷参数，默认1.0

    返回:
    电势值
    """
    return q / np.sqrt(a * a + z * z)

if __name__ == "__main__":
    # 测试单点电势
    print("原点(0,0,0)电势:", ring_potential_point(0, 0, 0))
    print("理论值: 1.0")
    print("z轴(0,0,1)电势:", ring_potential_point(0, 0, 1))
    print("理论值: {:.4f}".format(1 / np.sqrt(2)))

    # 测试网格形状
    y_test = np.array([[0, 1], [0, 1]])
    z_test = np.array([[0, 0], [1, 1]])
    v_test = ring_potential_grid(y_test, z_test)
    print("测试网格形状 - 输入y_grid:", y_test.shape, "z_grid:", z_test.shape)
    print("输出电势形状:", v_test.shape)

    # 可视化
    y = np.linspace(-3, 3, 100)
    z = np.linspace(-3, 3, 100)
    Y, Z = np.meshgrid(y, z)
    V = ring_potential_grid(Y, Z, x0=0.0)

    plt.figure(figsize=(8, 6))
    plt.contourf(Y, Z, V, levels=20, cmap='viridis')
    plt.colorbar(label='Potential')
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('均匀带电圆环yz平面等势线与电场分布')

    # 计算电场
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    dV_dy, dV_dz = np.gradient(V, dy, dz)
    Ey = -dV_dy
    Ez = -dV_dz

    # 绘制电场线
    plt.streamplot(Y, Z, Ey, Ez, color='white', density=1.5, linewidth=0.5)

    plt.show()

    plt.figure(figsize=(8, 6))
    plt.contourf(Y, Z, V, levels=20, cmap='viridis')
    plt.colorbar(label='Potential')
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('均匀带电圆环yz平面等势线与电场分布')

    # 计算电场
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    dV_dy, dV_dz = np.gradient(V, dy, dz)
    Ey = -dV_dy
    Ez = -dV_dz

    # 绘制电场线
    plt.streamplot(Y, Z, Ey, Ez, color='white', density=1.5, linewidth=0.5)

    plt.show()
