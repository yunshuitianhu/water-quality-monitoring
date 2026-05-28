"""水动力引擎调度器 — 检测 HEC-RAS 可用性, 选择引擎路径。"""

import os
from typing import List

from .config import YayaoConfig
from .cross_sections import CrossSection
from .state import HydroResult


def detect_engine() -> str:
    """检测可用的引擎类型。

    Returns:
        "hecras" — HEC-RAS COM 可用 (需管理员运行 Register batch)
        "pure_python" — 使用纯 Python Preissmann 求解器 (默认)
    """
    try:
        import win32com.client
        controller = win32com.client.Dispatch("RAS70.HECRASController")
        # 尝试获取版本号验证 COM 连通性
        _ = controller.HECRASVersion
        return "hecras"
    except Exception:
        pass

    return "pure_python"


def get_engine_info() -> str:
    engine = detect_engine()
    ras_installed = False
    from .hecras_bridge import is_hecras_available
    ras_installed = is_hecras_available()

    if engine == "hecras":
        return "HEC-RAS 7.0 (COM) — 使用 HEC-RAS 非恒定流引擎"
    elif ras_installed:
        return "纯 Python Preissmann 隐式格式求解器 (HEC-RAS 7.0 已检测到但 COM 未注册 — 以管理员运行 _Register_New_RAS_and_RASMapper_Files.bat)"
    else:
        return "纯 Python Preissmann 隐式格式求解器"


def run_hydrodynamic(
    config: YayaoConfig,
    cross_sections: List[CrossSection],
    upstream_flow_m3s: float = 6.9,
    downstream_stage_m: float = 0.83,
    tidal_amplitude_m: float = 0.8,
    tidal_period_h: float = 12.42,
    duration_h: float = 24.0,
    dt_s: float = 60.0,
    output_interval_min: float = 30.0,
) -> HydroResult:
    """运行水动力模拟 (自动选择引擎, HEC-RAS 失败时降级到纯 Python)。"""
    engine = detect_engine()

    if engine == "hecras":
        try:
            from .hecras_bridge import run_hecras_simulation
            return run_hecras_simulation(
                config, cross_sections,
                upstream_flow_m3s, downstream_stage_m,
                tidal_amplitude_m, tidal_period_h,
                duration_h, dt_s, output_interval_min,
            )
        except Exception as e:
            # HEC-RAS 失败 → 打印警告, 降级到纯 Python
            import warnings
            warnings.warn(f"HEC-RAS 运行失败, 降级到纯 Python: {e}")

    from .pure_py_hydro import solve_hydrodynamics
    return solve_hydrodynamics(
        config, cross_sections,
        upstream_flow_m3s, downstream_stage_m,
        tidal_amplitude_m, tidal_period_h,
        duration_h, dt_s, output_interval_min,
    )
