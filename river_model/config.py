"""雅瑶水道 (Yayao Waterway) 静态参数配置。

数据来源：《佛山市雅瑶水道建模资料汇总报告》(2026-05-26)
- 南海区政府官网、广东省交通运输厅、南方+、珠江水利科学研究院等
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class YayaoConfig:
    """雅瑶水道水动力+水质模型配置。"""

    # ---- 河道基本信息 ----
    river_name: str = "雅瑶水道"
    location: str = "广东省佛山市南海区"
    water_system: str = "北村水系"
    catchment_area_km2: float = 122.8

    # ---- 几何参数 ----
    length_km: float = 3.2
    length_m: float = 3200.0
    width_avg_m: float = 55.0
    width_range_m: Tuple[float, float] = (50.0, 220.0)
    depth_avg_m: float = 2.0
    depth_range_m: Tuple[float, float] = (1.8, 2.5)
    depth_max_m: float = 6.0
    cross_section_area_range_m2: Tuple[float, float] = (160.0, 200.0)
    bed_elevation_avg_m: float = -2.0
    bed_elevation_range_m: Tuple[float, float] = (-2.5, -1.5)
    bank_elevation_m: float = 1.5

    # ---- 水力学参数 ----
    mannings_n: float = 0.028
    mannings_n_range: Tuple[float, float] = (0.025, 0.035)
    design_discharge_m3s: float = 138.26
    avg_discharge_m3s: float = 6.9
    discharge_range_m3s: Tuple[float, float] = (6.3, 7.5)
    velocity_avg_ms: float = 0.06
    velocity_range_ms: Tuple[float, float] = (0.05, 0.07)
    longitudinal_slope: float = 0.0001

    # ---- 潮汐边界 ----
    tidal_range_m: float = 1.6
    tidal_range_limits_m: Tuple[float, float] = (1.35, 1.81)
    m2_period_h: float = 12.42
    upstream_structure: str = "北村水闸"
    downstream_connection: str = "珠江"
    # 北村水闸潮位特征 (1985国家高程基准)
    high_water_mean_m: float = 0.83
    low_water_mean_m: float = -0.52

    # ---- 水质基线 (V类) ----
    ammonia_nh3_mgL: float = 3.0
    dissolved_oxygen_mgL: float = 2.0
    cod_mgL: float = 60.0
    bod_cbod_mgL: float = 20.0
    phosphate_mgL: float = 0.4
    water_temp_C: float = 25.0

    # ---- GB3838-2002 V类标准限值 ----
    standard_nh3_v: float = 2.0
    standard_do_v: float = 2.0
    standard_cod_v: float = 40.0
    standard_bod_v: float = 10.0

    # ---- 数值参数 ----
    n_cross_sections: int = 50
    dt_default_s: float = 60.0
    dt_min_s: float = 10.0
    dt_max_s: float = 300.0
    preissmann_phi: float = 0.5
    preissmann_theta: float = 0.6
    max_newton_iter: int = 20
    newton_tolerance: float = 0.001  # 1mm 水位收敛容差 (HEC-RAS 典型值 ~3mm)

    # ---- 反应动力学参数 (20°C) ----
    k_cbod_decay: float = 0.25        # CBOD衰减速率 d⁻¹
    k_nitrification: float = 0.15     # 硝化速率 d⁻¹
    k_organic_n_hydrolysis: float = 0.20  # 有机氮水解 d⁻¹
    k_reaeration_base: float = 5.32   # Owens-Gibbs 复氧系数
    k_sediment_oxygen_demand: float = 1.0  # 底泥耗氧 g/m²/d
    theta_cbod: float = 1.047
    theta_nitrification: float = 1.072
    theta_reaeration: float = 1.024
    do_saturation_mgL: float = 8.25   # 25°C 饱和溶解氧

    # ---- 坐标参考 ----
    center_lat: float = 23.11
    center_lon: float = 113.14


DEFAULT_CONFIG = YayaoConfig()
