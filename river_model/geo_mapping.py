"""桩号 ↔ GPS 坐标双向映射。

从走航船监测数据提取 GPS 轨迹, 计算 Haversine 累积距离作为桩号,
建立 chainage ↔ (latitude, longitude) 的双向插值映射。
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from typing import Optional, Tuple


def _haversine(lon1: np.ndarray, lat1: np.ndarray,
               lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """向量化 Haversine 距离计算 (m)。"""
    R = 6371000.0
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2.0) ** 2)
    return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def build_chainage_gps_mapping(excel_path: str,
                                smoothing_sigma: float = 3.0,
                                interp_resolution_m: float = 50.0,
                                ) -> dict:
    """从走航船 Excel 数据建立 chainage ↔ GPS 双向映射。

    Args:
        excel_path: 走航数据 Excel 文件路径
        smoothing_sigma: 高斯平滑 sigma (点数), 值越大越平滑。默认 3.0
        interp_resolution_m: 插值网格分辨率 (m)。默认 50m 一个点

    Returns:
        dict:
            "chainage_grid": np.ndarray (m,) — 均匀间距的桩号网格
            "lat_grid": np.ndarray (m,) — 对应纬度
            "lon_grid": np.ndarray (m,) — 对应经度
            "chainage_raw": np.ndarray (n,) — 原始 GPS 点的累积距离
            "lat_raw": np.ndarray (n,) — 平滑后的原始点纬度
            "lon_raw": np.ndarray (n,) — 平滑后的原始点经度
            "total_length_m": float — 河道总长度
            "point_count": int — 原始 GPS 点数
            "_chainage_to_lat": interp1d — 正向插值器
            "_chainage_to_lon": interp1d — 正向插值器
    """
    # 1. 加载数据
    df = pd.read_excel(excel_path)

    # 列名映射 (与 loaders.py 一致)
    col_map = {
        '经度': 'lon', '纬度': 'lat',
        'longitude': 'lon', 'latitude': 'lat',
        'Longitude': 'lon', 'Latitude': 'lat',
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns},
              inplace=True)

    # 如果有时序列, 按时序排列; 否则按数据顺序
    if '时间' in df.columns:
        df['time'] = pd.to_datetime(df['时间'].astype(str).str.strip(), errors='coerce')
        df.dropna(subset=['time'], inplace=True)
        df.sort_values('time', inplace=True)
    if 'datetime' in df.columns:
        df.dropna(subset=['datetime'], inplace=True)
        df.sort_values('datetime', inplace=True)

    df.reset_index(drop=True, inplace=True)

    # 2. 提取坐标
    lats = df['lat'].values.astype(float)
    lons = df['lon'].values.astype(float)

    # 去除 NaN
    valid = ~(np.isnan(lats) | np.isnan(lons))
    lats = lats[valid]
    lons = lons[valid]

    if len(lats) < 10:
        raise ValueError(f"GPS 数据点不足 ({len(lats)} 个)。需要至少 10 个有效坐标点。")

    # 3. 高斯平滑 (消除船晃动噪声)
    lats_smooth = gaussian_filter1d(lats.astype(float), sigma=smoothing_sigma)
    lons_smooth = gaussian_filter1d(lons.astype(float), sigma=smoothing_sigma)

    # 4. 计算相邻点 Haversine 距离, 累积得到 chainage
    distances = _haversine(lons_smooth[:-1], lats_smooth[:-1],
                            lons_smooth[1:], lats_smooth[1:])
    chainage_raw = np.zeros(len(lats_smooth))
    chainage_raw[1:] = np.cumsum(distances)

    total_length_m = float(chainage_raw[-1])

    # 5. 构建均匀间距插值网格 (向前采样以供动画使用)
    n_grid = max(int(total_length_m / interp_resolution_m) + 1, len(chainage_raw))
    chainage_grid = np.linspace(0, total_length_m, n_grid)

    # 线性插值器
    lat_interp = interp1d(chainage_raw, lats_smooth, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    lon_interp = interp1d(chainage_raw, lons_smooth, kind='linear',
                           bounds_error=False, fill_value='extrapolate')

    lat_grid = lat_interp(chainage_grid)
    lon_grid = lon_interp(chainage_grid)

    # 确保插值范围不超出原始数据边界
    lat_grid = np.clip(lat_grid, lats_smooth.min() - 0.001, lats_smooth.max() + 0.001)
    lon_grid = np.clip(lon_grid, lons_smooth.min() - 0.001, lons_smooth.max() + 0.001)

    return {
        "chainage_grid": chainage_grid,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "chainage_raw": chainage_raw,
        "lat_raw": lats_smooth,
        "lon_raw": lons_smooth,
        "total_length_m": total_length_m,
        "point_count": len(lats_smooth),
        "_chainage_to_lat": lat_interp,
        "_chainage_to_lon": lon_interp,
    }


def chainage_to_gps(gps_mapping: dict, chainage_m: float) -> Tuple[float, float]:
    """将桩号转换为 GPS 坐标 (lat, lon)。"""
    lat = float(np.clip(gps_mapping["_chainage_to_lat"](chainage_m),
                         gps_mapping["lat_raw"].min() - 0.001,
                         gps_mapping["lat_raw"].max() + 0.001))
    lon = float(np.clip(gps_mapping["_chainage_to_lon"](chainage_m),
                         gps_mapping["lon_raw"].min() - 0.001,
                         gps_mapping["lon_raw"].max() + 0.001))
    return lat, lon


def chainage_array_to_gps(gps_mapping: dict,
                           chainage_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """将桩号数组批量转换为 GPS 坐标数组 (lats, lons)。"""
    lats = gps_mapping["_chainage_to_lat"](chainage_array)
    lons = gps_mapping["_chainage_to_lon"](chainage_array)
    lats = np.clip(lats,
                    gps_mapping["lat_raw"].min() - 0.001,
                    gps_mapping["lat_raw"].max() + 0.001)
    lons = np.clip(lons,
                    gps_mapping["lon_raw"].min() - 0.001,
                    gps_mapping["lon_raw"].max() + 0.001)
    return lats, lons


def get_gps_track_coords(gps_mapping: dict) -> list:
    """返回 Folium PolyLine 可用的坐标列表 [[lat, lon], ...]。
    使用插值网格, 保证轨迹平滑。
    """
    return list(zip(gps_mapping["lat_grid"].tolist(),
                    gps_mapping["lon_grid"].tolist()))


def get_map_center(gps_mapping: dict) -> Tuple[float, float]:
    """返回地图中心点 (lat, lon)。"""
    return (float(np.mean(gps_mapping["lat_raw"])),
            float(np.mean(gps_mapping["lon_raw"])))
