import numpy as np


def rate_3alpha(T: float) -> float:
    T8 = T / 1.0e8
    return 5.09e11 * (T8 ** (-3.0)) * np.exp(-44.027 / T8)


def finite_diff_dq_dT(T0: float, h: float = 1e-8) -> float:
    # 使用前向差分实现 dq/dT
    q0 = rate_3alpha(T0)
    q1 = rate_3alpha(T0 + h)
    return (q1 - q0) / h


def sensitivity_nu(T0: float, h: float = 1e-8) -> float:
    # 根据 nu = (T/q) * dq/dT 计算温度敏感性指数
    q = rate_3alpha(T0)
    dq_dT = finite_diff_dq_dT(T0, h)
    return (T0 / q) * dq_dT


def nu_table(T_values, h: float = 1e-8):
    # 返回 [(T, nu(T)), ...]
    return [(T, sensitivity_nu(T, h)) for T in T_values]
