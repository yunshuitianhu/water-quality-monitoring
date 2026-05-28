"""水质 ADR (Advection-Diffusion-Reaction) 求解器。

算子分裂法:
  1. 对流步: 显式迎风格式
  2. 扩散步: 显式中心差分
  3. 反应步: Runge-Kutta 4 阶

反应动力学 (简化 QUAL2E 型, 无藻类):
  CBOD: dL/dt = -(K_d + K_s)·L
  NH₃:  dN/dt = -β₁·N + S_sed
  DO:   dO/dt = -K_d·L - SOD/H + K_a·(O_sat - O)

纵向离散系数: Fischer 公式 (1979)
复氧系数: Owens-Gibbs (浅水河道)
"""

from typing import Optional

import numpy as np

from .config import YayaoConfig
from .state import HydroResult, WQResult


# ---- 反应动力学 ----

def _owens_gibbs_ka(u_ms: np.ndarray, H_m: np.ndarray) -> np.ndarray:
    """Owens-Gibbs 复氧系数 (d⁻¹)。

    适用范围: 浅水河道 (H < 3m), 低流速 (u < 0.5 m/s)。

    Ka = 5.32 * u^0.67 / H^1.85
    """
    u_safe = np.maximum(u_ms, 0.001)
    H_safe = np.maximum(H_m, 0.1)
    return 5.32 * u_safe ** 0.67 / H_safe ** 1.85


def _fischer_dispersion(u_ms: np.ndarray, B_m: np.ndarray,
                         H_m: np.ndarray, u_star: np.ndarray) -> np.ndarray:
    """Fischer 纵向离散系数 (m²/s)。

    K = 0.011 * u² * B² / (H * u*)
    """
    u_safe = np.maximum(u_ms, 0.001)
    B_safe = np.maximum(B_m, 1.0)
    H_safe = np.maximum(H_m, 0.1)
    u_star_safe = np.maximum(u_star, 0.001)
    K = 0.011 * u_safe ** 2 * B_safe ** 2 / (H_safe * u_star_safe)
    return np.clip(K, 0.5, 500.0)


# ---- 算子分裂 ----

def _advection_step(c: np.ndarray, u: np.ndarray, dx: float, dt: float) -> np.ndarray:
    """显式迎风格式求解对流方程 ∂c/∂t + u·∂c/∂x = 0。

    对每个断面 j:
      c_j^{n+1} = c_j^n - u_j·dt/dx * (c_j^n - c_{j-1}^n)   if u_j >= 0
      c_j^{n+1} = c_j^n - u_j·dt/dx * (c_{j+1}^n - c_j^n)   if u_j < 0
    """
    n = len(c)
    c_new = c.copy()
    for j in range(1, n - 1):
        if u[j] >= 0:
            c_new[j] = c[j] - u[j] * dt / dx * (c[j] - c[j - 1])
        else:
            c_new[j] = c[j] - u[j] * dt / dx * (c[j + 1] - c[j])
    # 上游边界: u >= 0 → 入流维持边界值; u < 0 → 零梯度
    if u[0] >= 0:
        c_new[0] = c[0]
    else:
        c_new[0] = c[1]
    # 下游边界: u >= 0 → 零梯度; u < 0 → 入流维持
    if u[-1] >= 0:
        c_new[-1] = c[-2]
    else:
        c_new[-1] = c[-1]
    return c_new


def _diffusion_step(c: np.ndarray, K: np.ndarray, dx: float, dt: float) -> np.ndarray:
    """显式中心差分求解扩散方程 ∂c/∂t = K·∂²c/∂x²。

    c_j^{n+1} = c_j^n + K_j·dt/dx² * (c_{j+1}^n - 2c_j^n + c_{j-1}^n)
    """
    n = len(c)
    c_new = c.copy()
    r = dt / (dx * dx)
    for j in range(1, n - 1):
        c_new[j] = c[j] + K[j] * r * (c[j + 1] - 2.0 * c[j] + c[j - 1])
    # 边界: 零通量
    c_new[0] = c_new[1]
    c_new[-1] = c_new[-2]
    return c_new


def _reaction_step(cbod: np.ndarray, nh3: np.ndarray, do_: np.ndarray,
                   H: np.ndarray, u: np.ndarray,
                   K_d: float, K_s: float, beta1: float, SOD: float,
                   O_sat: float, T_C: float, dt_s: float,
                   theta_d: float, theta_n: float, theta_rea: float) -> tuple:
    """RK4 求解反应步 dC/dt = f(C)。

    注意: 反应速率常数单位为 d⁻¹, dt_s 为秒, 需转换为天。
    """
    dt_day = dt_s / 86400.0  # 秒 → 天
    temp_fac_d = theta_d ** (T_C - 20.0)
    temp_fac_n = theta_n ** (T_C - 20.0)
    temp_fac_rea = theta_rea ** (T_C - 20.0)

    Ka = _owens_gibbs_ka(u, H)
    n = len(cbod)

    def rhs(L, N, O):
        dL = -(K_d * temp_fac_d + K_s) * L
        dN = -beta1 * temp_fac_n * N
        dO = (-K_d * temp_fac_d * L
              - SOD / np.maximum(H, 0.1)
              + Ka * temp_fac_rea * (O_sat - O))
        return dL, dN, dO

    # RK4 (使用 dt_day 因为速率常数为 d⁻¹)
    L1, N1, O1 = rhs(cbod, nh3, do_)
    L2, N2, O2 = rhs(cbod + 0.5 * dt_day * L1, nh3 + 0.5 * dt_day * N1, do_ + 0.5 * dt_day * O1)
    L3, N3, O3 = rhs(cbod + 0.5 * dt_day * L2, nh3 + 0.5 * dt_day * N2, do_ + 0.5 * dt_day * O2)
    L4, N4, O4 = rhs(cbod + dt_day * L3, nh3 + dt_day * N3, do_ + dt_day * O3)

    cbod_new = cbod + dt_day / 6.0 * (L1 + 2.0 * L2 + 2.0 * L3 + L4)
    nh3_new = nh3 + dt_day / 6.0 * (N1 + 2.0 * N2 + 2.0 * N3 + N4)
    do_new = do_ + dt_day / 6.0 * (O1 + 2.0 * O2 + 2.0 * O3 + O4)

    return (
        np.maximum(cbod_new, 0.0),
        np.maximum(nh3_new, 0.0),
        np.maximum(do_new, 0.0),
    )


# ---- 主求解器 ----

def solve_water_quality(
    config: YayaoConfig,
    hydro: HydroResult,
    pollutant_type: str = "ammonia",
    load_kg: float = 50.0,
    chainage_m: float = 1600.0,
    duration_min: float = 30.0,
    simulation_hours: float = 24.0,
    dt_s: float = 60.0,
) -> WQResult:
    """运行水质 ADR 模拟 (含污染事件)。

    Args:
        config: 雅瑶水道配置
        hydro: 水动力结果 (由 solve_hydrodynamics 产出)
        pollutant_type: "ammonia" | "cbod" | "cod" | "conservative"
        load_kg: 污染物总质量 (kg)
        chainage_m: 污染源位置 (m, 从上游起算)
        duration_min: 排放持续时间 (min)
        simulation_hours: 模拟总时长 (h)
        dt_s: 水质时间步长 (s), 建议 ≤ 水动力步长

    Returns:
        WQResult 包含各组分浓度时间序列
    """
    n_cs = len(hydro.chainage)
    chainage = hydro.chainage
    t_hydro = hydro.t * 3600.0  # 转为秒
    duration_sec = duration_min * 60.0

    # ---- 查找污染源位置 (最近断面) ----
    source_idx = int(np.argmin(np.abs(chainage - chainage_m)))

    # ---- 参数 ----
    K_d = config.k_cbod_decay        # d⁻¹
    K_s = 0.05                        # CBOD 沉降速率 d⁻¹
    beta1 = config.k_nitrification   # 硝化速率 d⁻¹
    SOD = config.k_sediment_oxygen_demand  # g/m²/d
    O_sat = config.do_saturation_mgL
    T_C = config.water_temp_C

    theta_d = config.theta_cbod
    theta_n = config.theta_nitrification
    theta_rea = config.theta_reaeration

    # ---- 计算离散系数 ----
    # 对每个断面估算 Fischer 离散系数
    g = 9.81
    B_array = np.array([55.0] * n_cs)  # 平均河宽
    K_array = np.ones(n_cs) * 5.0      # 默认 5 m²/s

    # ---- 初始浓度场 ----
    cbod = np.ones(n_cs) * config.bod_cbod_mgL
    nh3 = np.ones(n_cs) * config.ammonia_nh3_mgL
    do_ = np.ones(n_cs) * config.dissolved_oxygen_mgL

    # ---- 时间推进 ----
    total_steps = int(simulation_hours * 3600.0 / dt_s)
    n_outputs = max(int(total_steps / 60) + 1, 2)  # 每分钟输出一次
    output_interval = max(total_steps // (n_outputs - 1), 1)

    n_out = total_steps // output_interval + 1
    t_out = np.zeros(n_out)
    cbod_out = np.zeros((n_out, n_cs))
    nh3_out = np.zeros((n_out, n_cs))
    do_out = np.zeros((n_out, n_cs))
    out_idx = 0

    # 保存初始状态
    t_out[0] = 0.0
    cbod_out[0, :] = cbod
    nh3_out[0, :] = nh3
    do_out[0, :] = do_
    out_idx = 1

    dx = chainage[1] - chainage[0] if n_cs > 1 else 64.0
    # 使用水动力结果中平均或最接近时间点的流速
    hydro_dt = np.mean(np.diff(t_hydro)) if len(t_hydro) > 1 else dt_s

    # 质量注入速率 (kg/s → 浓度 mg/L)
    # load_kg over duration_sec → mass_rate g/s
    mass_rate_gs = load_kg * 1000.0 / duration_sec if duration_sec > 0 else 0.0
    # 注入浓度增量: ΔC = mass_rate / Q_at_source (mg/L)
    # 使用水动力结果中该断面的平均流量

    for step in range(1, total_steps + 1):
        t_current = step * dt_s

        # ---- 从水动力结果线性插值当前时刻的 u, H, A, Q ----
        if len(t_hydro) > 1:
            hydro_idx_frac = np.interp(t_current, t_hydro,
                                        np.arange(len(t_hydro)))
            hydro_lo = int(np.floor(hydro_idx_frac))
            hydro_hi = min(hydro_lo + 1, len(t_hydro) - 1)
            frac = hydro_idx_frac - hydro_lo
            # 插值流速 (保留方向, 不用 np.abs)
            u_lo = hydro.velocity[hydro_lo, :]
            u_hi = hydro.velocity[hydro_hi, :]
            u_now = (1.0 - frac) * u_lo + frac * u_hi
            # 插值过流面积
            if hydro.area is not None:
                a_lo = hydro.area[hydro_lo, :]
                a_hi = hydro.area[hydro_hi, :]
                A_now = (1.0 - frac) * a_lo + frac * a_hi
            else:
                A_now = np.ones(n_cs) * config.width_avg_m * config.depth_avg_m
            # 插值流量 (保留方向)
            q_lo = hydro.discharge[hydro_lo, :]
            q_hi = hydro.discharge[hydro_hi, :]
            Q_now = (1.0 - frac) * q_lo + frac * q_hi
        else:
            # 只有稳态结果, 直接用
            u_now = hydro.velocity[0, :]
            A_now = hydro.area[0, :] if hydro.area is not None else np.ones(n_cs) * config.width_avg_m * config.depth_avg_m
            Q_now = hydro.discharge[0, :]

        # 水深: 面积 / 河宽
        H_now = np.maximum(A_now / config.width_avg_m,
                            config.depth_range_m[0] * 0.5)

        # 污染源注入
        # mass_rate [g/s] / flow [m³/s] = [mg/L]  (1 g/m³ = 1 mg/L, 1 m³/s = 1000 L/s)
        if t_current <= duration_sec and pollutant_type != "conservative":
            Q_src = max(abs(Q_now[source_idx]), 0.1)
            delta_c = mass_rate_gs / Q_src  # mg/L
            if pollutant_type == "ammonia":
                nh3[source_idx] += delta_c * dt_s / duration_sec
            elif pollutant_type in ("cbod", "cod"):
                cbod[source_idx] += delta_c * dt_s / duration_sec

        # ---- 算子分裂 ----
        # 子步长可能需要更小以满足 CFL (u*dt/dx < 1)
        n_sub = max(1, int(np.max(u_now) * dt_s / dx) + 1)
        dt_sub = dt_s / n_sub

        for _ in range(n_sub):
            # 1. 对流
            cbod = _advection_step(cbod, u_now, dx, dt_sub)
            nh3 = _advection_step(nh3, u_now, dx, dt_sub)
            do_ = _advection_step(do_, u_now, dx, dt_sub)

            # 2. 扩散
            cbod = _diffusion_step(cbod, K_array, dx, dt_sub)
            nh3 = _diffusion_step(nh3, K_array, dx, dt_sub)
            do_ = _diffusion_step(do_, K_array, dx, dt_sub)

            # 3. 反应
            cbod, nh3, do_ = _reaction_step(
                cbod, nh3, do_, H_now, u_now,
                K_d, K_s, beta1, SOD, O_sat, T_C, dt_sub,
                theta_d, theta_n, theta_rea,
            )

        # ---- 输出 ----
        if step % output_interval == 0 and out_idx < n_out:
            t_out[out_idx] = t_current / 3600.0
            cbod_out[out_idx, :] = cbod
            nh3_out[out_idx, :] = nh3
            do_out[out_idx, :] = do_
            out_idx += 1

    # 确保最后一个时间步被写下
    if out_idx < n_out:
        t_out[out_idx] = total_steps * dt_s / 3600.0
        cbod_out[out_idx, :] = cbod
        nh3_out[out_idx, :] = nh3
        do_out[out_idx, :] = do_
        out_idx += 1

    t_out = t_out[:out_idx]
    cbod_out = cbod_out[:out_idx, :]
    nh3_out = nh3_out[:out_idx, :]
    do_out = do_out[:out_idx, :]

    return WQResult(
        t=t_out,
        chainage=chainage,
        cbod=cbod_out,
        ammonia=nh3_out,
        dissolved_oxygen=do_out,
        params={
            "pollutant_type": pollutant_type,
            "load_kg": load_kg,
            "chainage_m": chainage_m,
            "source_section_id": source_idx,
            "duration_min": duration_min,
            "simulation_hours": simulation_hours,
        },
    )
