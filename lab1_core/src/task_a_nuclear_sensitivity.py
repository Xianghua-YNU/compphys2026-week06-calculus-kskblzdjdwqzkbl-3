import numpy as np


def rate_3alpha(T: float) -> float:
    """
    计算恒星中3-α反应的温度相关反应率 q(T)
    参数:
        T: 温度，单位为开尔文(K)，需大于0
    返回:
        q(T) 的计算值
    异常:
        当T≤0时抛出ValueError，避免数值溢出/无意义计算
    """
    if T <= 0:
        raise ValueError(f"温度T必须大于0，当前输入为{T} K")
    T8 = T / 1.0e8
    q=5.09e11 * (T8 ** (-3.0)) * np.exp(-44.027 / T8)

    # 检查数值溢出（常见错误点）
    if np.isinf(q) or np.isnan(q):
        raise OverflowError(f"计算q(T)时发生数值溢出，T={T} K")
    
    return q



def finite_diff_dq_dT(T0: float, h: float = 1e-8) -> float:
    # TODO A1: 使用前向差分实现 dq/dT
     """
    使用前向差分法计算 dq/dT 在T0处的导数
    公式: dq/dT ≈ [q(T0+h*T0) - q(T0)] / (h*T0) （h是相对增量，非绝对增量，常见错误点）
    参数:
        T0: 参考温度，单位K
        h: 相对步长（默认1e-8，避免步长过大/过小导致误差）
    返回:
        dq/dT 在T0处的导数值
    """
    # 计算增量：h是相对增量，需乘以T0（常见错误点：避免把h当作绝对增量）
    delta_T = h * T0
    T1 = T0 + delta_T
    
    # 计算q(T0)和q(T1)
    q0 = rate_3alpha(T0)
    q1 = rate_3alpha(T1)
    
    # 前向差分计算导数
    dq_dT = (q1 - q0) / delta_T
    return dq_dT


def sensitivity_nu(T0: float, h: float = 1e-8) -> float:
    # TODO A2: 根据 nu = (T/q) * dq/dT 计算温度敏感性指数
    """
    计算温度敏感性指数 ν = (T0/q0) * (dq/dT)|T0
    参数:
        T0: 参考温度，单位K
        h: 相对步长（传递给有限差分函数）
    返回:
        温度敏感性指数ν
    """
    q0 = rate_3alpha(T0)
    dq_dT = finite_diff_dq_dT(T0, h)
    
    # 核心公式：nu = (T/q) * dq/dT （注意q用T0处的值，非T0+ΔT，常见错误点）
    nu = (T0 / q0) * dq_dT
    return nu


def nu_table(T_values, h: float = 1e-8):
    # TODO A3: 返回 [(T, nu(T)), ...]
    """
    生成温度-敏感性指数对照表
    参数:
        T_values: 温度列表/数组（单位K）
        h: 相对步长
    返回:
        列表，每个元素为元组 (T, nu(T))
    """
    result = []
    for T in T_values:
        try:
            nu = sensitivity_nu(T, h)
            result.append((T, nu))
        except (ValueError, OverflowError) as e:
            print(f"警告：温度{T} K计算失败: {e}")
            result.append((T, np.nan))  # 失败时标记为nan
    return result

    # ------------------- 测试代码（验证结果符合自检要求） -------------------
if __name__ == "__main__":
    # 测试温度范围：覆盖1e8 K（参考点）、高温区（验证趋势）
    test_T = [1e8, 1.2e8, 1.5e8, 2e8, 5e8, 1e9]
    
    # 生成对照表
    nu_results = nu_table(test_T)
    
    # 打印结果
    print("温度(K) | 敏感性指数ν")
    print("---------------------")
    for T, nu in nu_results:
        print(f"{T:.1e} | {nu:.2f}")
