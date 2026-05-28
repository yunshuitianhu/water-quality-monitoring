"""河道模型状态管理。"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import YayaoConfig, DEFAULT_CONFIG
from .cross_sections import CrossSection


@dataclass
class HydroResult:
    """一次水动力模拟的结果。"""
    t: np.ndarray                     # 时间序列 (s), shape (nt,)
    chainage: np.ndarray             # 桩号 (m), shape (ncs,)
    water_level: np.ndarray          # 水位 (m), shape (nt, ncs)
    discharge: np.ndarray            # 流量 (m³/s), shape (nt, ncs)
    velocity: np.ndarray             # 流速 (m/s), shape (nt, ncs)
    area: np.ndarray                 # 过水面积 (m²), shape (nt, ncs)
    params: dict = field(default_factory=dict)  # 运行时参数


@dataclass
class WQResult:
    """一次水质模拟的结果。"""
    t: np.ndarray                    # 时间序列 (s), shape (nt,)
    chainage: np.ndarray            # 桩号 (m), shape (ncs,)
    cbod: Optional[np.ndarray] = None    # (nt, ncs)
    ammonia: Optional[np.ndarray] = None # (nt, ncs)
    nitrate: Optional[np.ndarray] = None  # (nt, ncs)
    dissolved_oxygen: Optional[np.ndarray] = None  # (nt, ncs)
    phosphate: Optional[np.ndarray] = None  # (nt, ncs)
    params: dict = field(default_factory=dict)


@dataclass
class RiverModelState:
    """河道模型全局状态。"""
    config: YayaoConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    cross_sections: List[CrossSection] = field(default_factory=list)
    hydro_result: Optional[HydroResult] = None
    wq_result: Optional[WQResult] = None
    scenarios: Dict[str, dict] = field(default_factory=dict)
    engine_type: str = "pure_python"
    model_initialized: bool = False

    def has_hydro(self) -> bool:
        return self.hydro_result is not None

    def has_wq(self) -> bool:
        return self.wq_result is not None
