"""水质监测溯源 MCP Server — 12 个工具供 Claude Code 调用。"""

import json

import numpy as np
from mcp.server.fastmcp import FastMCP

from .analysis import (
    analyze_video_frames,
    build_comprehensive_summary,
    cross_validate_video_with_monitoring,
    data_summary,
    find_black_spots,
)
from .charts import (
    generate_correlation_heatmap,
    generate_cross_validation_chart,
    generate_time_series,
    generate_trace_map,
)
from .config import resolve_api_key
from .geo_tools import (
    get_satellite_image,
    reverse_geocode,
    search_nearby_pollution_sources,
)
from .loaders import extract_video, load_data
from .national_station import get_national_station_data
from .report import generate_report
from .state import WaterQualityState, _get_river_state

mcp = FastMCP("water-quality-mcp")
state = WaterQualityState()


# ---------- Tool 1: load_data ----------
@mcp.tool(name="load_data", annotations={"readOnlyHint": False, "destructiveHint": False})
def tool_load_data(file_bytes: str, filename: str) -> str:
    """加载并清洗走航监测数据 (CSV/Excel)。传入 base64 编码的文件内容和文件名。"""
    df, report = load_data(file_bytes, filename)
    if df is not None:
        state.df = df
    return report


# ---------- Tool 2: get_data_summary ----------
@mcp.tool(name="get_data_summary", annotations={"readOnlyHint": True})
def tool_get_data_summary() -> str:
    """返回已加载数据的统计摘要（记录数、时间范围、氨氮/溶解氧/浊度/pH/COD 均值与极值）。"""
    return data_summary(state.df)


# ---------- Tool 3: find_black_spots ----------
@mcp.tool(name="find_black_spots", annotations={"readOnlyHint": False, "destructiveHint": False})
def tool_find_black_spots() -> str:
    """基于《城市黑臭水体整治工作指南》(2015)和GB3838-2002检测水质黑点，DBSCAN聚类后返回黑臭等级和主要污染物。"""
    result, black_spots = find_black_spots(state.df)
    if black_spots is not None:
        state.black_spots = black_spots
    return result


# ---------- Tool 4: generate_time_series ----------
@mcp.tool(name="generate_time_series", annotations={"readOnlyHint": False, "destructiveHint": False})
def tool_generate_time_series(output_path: str = "time_series.png", font_path: str | None = None) -> str:
    """生成 4 面板时间序列图（氨氮/溶解氧/浊度/pH），黑点按严重等级着色。"""
    return generate_time_series(state, output_path, font_path)


# ---------- Tool 5: generate_trace_map ----------
@mcp.tool(name="generate_trace_map", annotations={"readOnlyHint": False, "destructiveHint": False})
def tool_generate_trace_map(
    output_path: str = "trace_map.html",
    amap_key: str | None = None,
    cloud_token: str | None = None,
) -> str:
    """生成交互式 folium 溯源地图（含底图、热力图、聚类范围、污染源标记、严重程度图例）。"""
    amap_key = resolve_api_key("AMAP_KEY", amap_key)
    cloud_token = resolve_api_key("CLOUD_TOKEN", cloud_token)
    return generate_trace_map(state, output_path, amap_key, cloud_token)


# ---------- Tool 6: get_satellite_image ----------
@mcp.tool(name="get_satellite_image", annotations={"readOnlyHint": True})
def tool_get_satellite_image(lat: float, lon: float, zoom: int = 16, cloud_token: str | None = None) -> str:
    """获取星图地球卫星影像瓦片的状态信息。经纬度自动转换为 XYZ 瓦片编号。"""
    cloud_token = resolve_api_key("CLOUD_TOKEN", cloud_token)
    return get_satellite_image(lat, lon, zoom, cloud_token)


# ---------- Tool 7: search_nearby_pollution_sources ----------
@mcp.tool(name="search_nearby_pollution_sources", annotations={"readOnlyHint": True})
def tool_search_nearby_pollution_sources(
    lat: float, lon: float, radius: int = 500, amap_key: str | None = None
) -> str:
    """搜索周边污染源（工厂/养殖场/排污口/化工厂），按距离排序。使用高德 POI 搜索 API。"""
    amap_key = resolve_api_key("AMAP_KEY", amap_key)
    if not amap_key:
        return json.dumps({"error": "未配置高德 API Key (AMAP_KEY)"}, ensure_ascii=False)
    pois = search_nearby_pollution_sources(lat, lon, amap_key, radius)
    return json.dumps({"搜索中心": f"{lat:.6f},{lon:.6f}", "半径": f"{radius}m", "结果": pois[:15]}, ensure_ascii=False)


# ---------- Tool 8: reverse_geocode ----------
@mcp.tool(name="reverse_geocode", annotations={"readOnlyHint": True})
def tool_reverse_geocode(lat: float, lon: float, amap_key: str | None = None) -> str:
    """逆地理编码：将经纬度转换为地址、省份和周边 POI。使用高德逆地理编码 API。"""
    amap_key = resolve_api_key("AMAP_KEY", amap_key)
    if not amap_key:
        return json.dumps({"error": "未配置高德 API Key (AMAP_KEY)"}, ensure_ascii=False)
    return reverse_geocode(lat, lon, amap_key)


# ---------- Tool 9: generate_report ----------
@mcp.tool(name="generate_report", annotations={"readOnlyHint": False, "destructiveHint": False})
def tool_generate_report(summary: str, api_key: str | None = None) -> str:
    """基于 GB 3838-2002 标准生成水环境溯源报告。调用 DeepSeek API。"""
    api_key = resolve_api_key("DEEPSEEK_API_KEY", api_key)
    if not api_key:
        return json.dumps({"error": "未配置 DeepSeek API Key (DEEPSEEK_API_KEY)"}, ensure_ascii=False)
    return generate_report(api_key, summary)


# ---------- Tool 10: extract_video ----------
@mcp.tool(name="extract_video", annotations={"readOnlyHint": False, "destructiveHint": False})
def tool_extract_video(file_bytes: str, interval: int = 10) -> str:
    """从视频文件中按间隔抽帧。传入 base64 编码的视频文件。"""
    frames, report = extract_video(file_bytes, interval)
    state.video_frames = frames
    return report


# ---------- Tool 11: analyze_video_frames ----------
@mcp.tool(name="analyze_video_frames", annotations={"readOnlyHint": True})
def tool_analyze_video_frames() -> str:
    """分析已抽取的视频帧，评估浑浊度和颜色异常。"""
    return analyze_video_frames(state.video_frames)


# ---------- Tool 12: cross_validate_video ----------
@mcp.tool(name="cross_validate_video", annotations={"readOnlyHint": True})
def tool_cross_validate_video(time_offset_sec: int = 0, time_window_sec: int = 30) -> str:
    """将视频帧视觉特征与走航监测数据交叉印证。

    计算 Laplacian 方差/HSV 饱和度/亮度 与 浊度/氨氮/DO/COD 的 Pearson 相关系数，
    以及视频黑臭判断与传感器黑臭判定的一致性。

    Args:
        time_offset_sec: 视频第0秒相对第一条监测记录的时间偏移(秒), 正=视频滞后
        time_window_sec: 匹配窗口宽度(秒), 取窗口内监测数据均值
    """
    result = cross_validate_video_with_monitoring(state, time_offset_sec, time_window_sec)
    state.cross_validation_result = json.loads(result) if isinstance(result, str) else result
    return result


# ---------- Tool 13: generate_cross_validation_chart ----------
@mcp.tool(name="generate_cross_validation_chart", annotations={"readOnlyHint": False, "destructiveHint": False})
def tool_generate_cross_validation_chart(time_offset_sec: int = 0, output_path: str = "cross_validation.png") -> str:
    """生成视频-传感器交叉印证双面板时间序列对比图。

    上: Laplacian方差 vs 浊度 | 下: HSV亮度 vs 氨氮
    """
    return generate_cross_validation_chart(state, output_path, time_offset_sec)


# ---------- Tool 14: generate_correlation_heatmap ----------
@mcp.tool(name="generate_correlation_heatmap", annotations={"readOnlyHint": False, "destructiveHint": False})
def tool_generate_correlation_heatmap(output_path: str = "correlation_heatmap.png") -> str:
    """生成视频指标 × 传感器指标 Pearson 相关系数热力图。

    需先调用 cross_validate_video。
    """
    if state.cross_validation_result is None:
        return "请先调用 cross_validate_video 进行交叉验证"
    return generate_correlation_heatmap(state.cross_validation_result, output_path)


# ---------- Tool 15: get_comprehensive_summary ----------
@mcp.tool(name="get_comprehensive_summary", annotations={"readOnlyHint": True})
def tool_get_comprehensive_summary() -> str:
    """聚合所有分析结果为统一 JSON。

    包含数据概览、黑臭检测(含聚类详情和Top5最严重点位)、空间分析、交叉印证、河道模型。
    LLM 可借此获得完整数据图景, 无需逐工具调用。
    """
    return build_comprehensive_summary(state, _get_river_state())


# ---------- Tool 12: get_national_station_data ----------
@mcp.tool(name="get_national_station_data", annotations={"readOnlyHint": True})
def tool_get_national_station_data(province: str = "", keyword: str = "") -> str:
    """获取国控水质站点数据。按省份或关键词筛选。"""
    return get_national_station_data(province, keyword)


# ---------- Tool 13: init_river_model ----------
@mcp.tool(name="init_river_model", annotations={"readOnlyHint": False, "destructiveHint": False})
def tool_init_river_model(n_cross_sections: int = 50, mannings_n: float = 0.028) -> str:
    """初始化雅瑶水道一维水动力-水质模型。加载河道几何断面、曼宁糙率、潮汐边界条件。"""
    from river_model.tools import tool_river_init
    return tool_river_init(_get_river_state(), n_cross_sections, mannings_n)


# ---------- Tool 14: run_hydrodynamic_simulation ----------
@mcp.tool(name="run_hydrodynamic_simulation", annotations={"readOnlyHint": False, "destructiveHint": False})
def tool_run_hydrodynamic_simulation(
    upstream_flow_m3s: float = 6.9,
    downstream_stage_m: float = 0.83,
    tidal_amplitude_m: float = 0.8,
    tidal_period_h: float = 12.42,
    duration_h: float = 24.0,
    dt_s: float = 60.0,
) -> str:
    """运行雅瑶水道一维水动力模拟(Saint-Venant方程)。返回水位/流速/流量统计。"""
    from river_model.tools import tool_river_hydro_simulation
    return tool_river_hydro_simulation(
        _get_river_state(),
        upstream_flow_m3s, downstream_stage_m,
        tidal_amplitude_m, tidal_period_h,
        duration_h, dt_s,
    )


# ---------- Tool 15: simulate_pollution_event ----------
@mcp.tool(name="simulate_pollution_event", annotations={"readOnlyHint": False, "destructiveHint": False})
def tool_simulate_pollution_event(
    chainage_m: float = 1600.0,
    pollutant_type: str = "ammonia",
    load_kg: float = 50.0,
    duration_min: float = 30.0,
    simulation_hours: float = 24.0,
) -> str:
    """在雅瑶水道指定位置模拟污染事件,运行ADR水质模型。返回峰值浓度/影响范围/达标时间。"""
    from river_model.tools import tool_river_pollution_event
    return tool_river_pollution_event(
        _get_river_state(),
        chainage_m, pollutant_type, load_kg, duration_min, simulation_hours,
    )


# ---------- Tool 16: get_concentration_profile ----------
@mcp.tool(name="get_concentration_profile", annotations={"readOnlyHint": True})
def tool_get_concentration_profile(time_h: float = 6.0, constituent: str = "ammonia") -> str:
    """查询雅瑶水道指定时刻的沿程浓度剖面。返回各断面浓度和超标统计。"""
    from river_model.tools import tool_river_concentration_profile
    return tool_river_concentration_profile(_get_river_state(), time_h, constituent)


# ---------- Tool 17: compare_scenarios ----------
@mcp.tool(name="compare_scenarios", annotations={"readOnlyHint": True})
def tool_compare_scenarios(scenario1_name: str = "", scenario2_name: str = "") -> str:
    """对比雅瑶水道两个已保存的模拟情景。"""
    from river_model.tools import tool_river_compare_scenarios
    return tool_river_compare_scenarios(_get_river_state(), scenario1_name, scenario2_name)


# ---------- Tool 18: export_river_model_data ----------
@mcp.tool(name="export_river_model_data", annotations={"readOnlyHint": True})
def tool_export_river_model_data(format: str = "json", scenario_name: str = "latest") -> str:
    """导出雅瑶水道模拟结果数据(JSON/CSV)。"""
    from river_model.tools import tool_river_export_data
    return tool_river_export_data(_get_river_state(), format, scenario_name)


# ---------- Tool 19: generate_river_map_animation ----------
@mcp.tool(name="generate_river_map_animation", annotations={"readOnlyHint": False})
def tool_generate_river_map_animation(
    constituent: str = "ammonia",
    output_path: str = "river_pollution_animation.html",
    source_chainage_m: float = 0.0,
    excel_path: str = "苗圃杯/无人船智能水质监测分析和因果溯源样本数据/河道巡航走航样本数据.xlsx",
    dual_view: bool = True,
) -> str:
    """生成雅瑶水道污染扩散地图动画HTML。

    基于走航船 GPS 轨迹和 ADR 污染模拟结果, 生成带时间轴的污染扩散地图动画。
    使用 GB3838-2002 标准色标, 展示污染物浓度沿河道随时间扩散的过程。
    必须先运行 simulate_pollution_event。
    """
    from river_model.tools import tool_river_map_animation
    return tool_river_map_animation(
        _get_river_state(),
        excel_path=excel_path,
        constituent=constituent,
        output_path=output_path,
        source_chainage_m=source_chainage_m if source_chainage_m > 0 else None,
        dual_view=dual_view,
    )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
