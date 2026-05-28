"""雅瑶水道断面几何生成。

基于建模报告中的断面参数，生成梯形断面序列。
断面上游至下游分布，间距 ~64m (3.2km / 50断面)。
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class CrossSection:
    """单个河道断面定义。

    Attributes:
        id: 断面编号 (0 = 上游北村水闸)
        chainage_m: 距上游起点距离 (m)
        stations: 横向偏移-高程对 (n_points, 2), 列 [offset_m, elevation_m]
        mannings_n_left: 左滩地糙率
        mannings_n_channel: 主槽糙率
        mannings_n_right: 右滩地糙率
    """
    id: int
    chainage_m: float
    stations: np.ndarray         # shape (n, 2): [lateral_m, elevation_m]
    mannings_n_left: float = 0.028
    mannings_n_channel: float = 0.028
    mannings_n_right: float = 0.035

    @property
    def bottom_elevation(self) -> float:
        return float(np.min(self.stations[:, 1]))

    @property
    def channel_width(self) -> float:
        return float(np.max(self.stations[:, 0]) - np.min(self.stations[:, 0]))

    def hydraulic_radius(self, water_level: float) -> float:
        """给定水位下的水力半径 R = A / P。"""
        A = self.flow_area(water_level)
        P = self.wetted_perimeter(water_level)
        return A / P if P > 1e-6 else 0.0

    def flow_area(self, water_level: float) -> float:
        """梯形积分计算过水面积。"""
        s = self.stations
        depth = np.maximum(0.0, water_level - s[:, 1])
        # 梯形积分: sum of (depth[i] + depth[i+1]) / 2 * dx[i]
        dx = np.diff(s[:, 0])
        area = np.sum((depth[:-1] + depth[1:]) / 2.0 * dx)
        return float(area)

    def wetted_perimeter(self, water_level: float) -> float:
        """湿周 (近似)。"""
        width = self.top_width(water_level)
        depth = water_level - self.bottom_elevation
        return float(width + 2.0 * max(0.0, depth))

    def top_width(self, water_level: float) -> float:
        """水面宽度。"""
        s = self.stations
        wet = water_level > s[:, 1]
        if wet.sum() < 2:
            return 0.0
        left_idx = np.argmax(wet)
        right_idx = len(wet) - 1 - np.argmax(wet[::-1])
        return float(s[right_idx, 0] - s[left_idx, 0])

    def conveyance(self, water_level: float) -> float:
        """计算 K = A * R^(2/3) / n。"""
        A = self.flow_area(water_level)
        R = self.hydraulic_radius(water_level)
        if A < 1e-6 or R < 1e-6:
            return 0.0
        return float(A * R ** (2.0 / 3.0) / self.mannings_n_channel)


def generate_yayao_cross_sections(
    n_total: int = 50,
    channel_length_m: float = 3200.0,
    bed_elevation_avg_m: float = -2.0,
    bed_slope: float = 0.0001,
    bottom_width_m: float = 40.0,
    bank_elevation_m: float = 1.5,
    side_slope: float = 2.0,       # 水平:垂直 = 2:1
    mannings_n: float = 0.028,
) -> List[CrossSection]:
    """生成雅瑶水道理想化梯形断面序列。

    上游 (id=0) 河底高程略高 (bed - slope*length/2)，
    下游河底高程略低 (bed + slope*length/2)，形成缓坡。

    Args:
        n_total: 断面总数
        channel_length_m: 河段总长 (m)
        bed_elevation_avg_m: 平均河底高程 (m, 1985高程基准)
        bed_slope: 河底纵向坡降
        bottom_width_m: 河底宽度 (m)
        bank_elevation_m: 堤岸高程 (m)
        side_slope: 边坡系数 (水平/垂直)
        mannings_n: 主槽糙率

    Returns:
        断面列表，从上游到下游排列
    """
    dx = channel_length_m / (n_total - 1)
    sections = []

    for i in range(n_total):
        chainage = i * dx
        # 河底高程沿程线性变化
        bed_el = bed_elevation_avg_m - bed_slope * (channel_length_m / 2.0 - chainage)

        # 梯形断面: 5个控制点
        half_bottom = bottom_width_m / 2.0
        bank_offset = half_bottom + side_slope * (bank_elevation_m - bed_el)
        stations = np.array([
            [-bank_offset, bank_elevation_m],  # 左岸
            [-half_bottom, bed_el],             # 左河底拐点
            [0.0,          bed_el],             # 河底中心
            [half_bottom,  bed_el],             # 右河底拐点
            [bank_offset,  bank_elevation_m],   # 右岸
        ], dtype=np.float64)

        sections.append(CrossSection(
            id=i,
            chainage_m=chainage,
            stations=stations,
            mannings_n_left=mannings_n + 0.005,
            mannings_n_channel=mannings_n,
            mannings_n_right=mannings_n + 0.005,
        ))

    return sections
