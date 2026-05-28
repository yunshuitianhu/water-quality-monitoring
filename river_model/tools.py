"""河道模型工具包装函数 — 供 app.py 和 MCP server 调用。

每个函数返回 JSON 字符串, 遵循现有的 tool 返回格式 (≤2000 字符摘要)。
"""

import json
import os

import numpy as np

from .config import YayaoConfig, DEFAULT_CONFIG
from .cross_sections import generate_yayao_cross_sections
from .hydro_engine import run_hydrodynamic, get_engine_info
from .state import RiverModelState
from .visualization import (
    plot_water_level_profile,
    plot_discharge_profile,
    plot_concentration_contour,
    plot_concentration_profile,
    plot_concentration_timeseries,
    plot_hydro_wq_dashboard,
)
from .wq_engine import solve_water_quality


# ---- Tool 1: 初始化模型 ----

def tool_river_init(
    state: RiverModelState,
    n_cross_sections: int = 50,
    mannings_n: float = 0.028,
) -> str:
    """初始化雅瑶水道一维水动力-水质模型。

    加载河道几何断面、糙率、潮汐边界条件。
    """
    config = YayaoConfig(
        n_cross_sections=n_cross_sections,
        mannings_n=mannings_n,
    )
    cs_list = generate_yayao_cross_sections(
        n_total=n_cross_sections,
        channel_length_m=config.length_m,
        bed_elevation_avg_m=config.bed_elevation_avg_m,
        bed_slope=config.longitudinal_slope,
        bottom_width_m=40.0,
        mannings_n=mannings_n,
    )
    state.config = config
    state.cross_sections = cs_list
    state.model_initialized = True
    state.engine_type = get_engine_info()

    engine_label = "HEC-RAS" if "HEC-RAS" in state.engine_type else "纯Python Preissmann"

    return json.dumps({
        "状态": "模型已初始化",
        "河道名称": config.river_name,
        "位置": config.location,
        "河段长度_km": config.length_km,
        "断面数": len(cs_list),
        "断面间距_m": round(config.length_m / (len(cs_list) - 1), 1) if len(cs_list) > 1 else 0,
        "平均河宽_m": config.width_avg_m,
        "平均水深_m": config.depth_avg_m,
        "曼宁糙率": mannings_n,
        "下游潮差_m": config.tidal_range_m,
        "平均流量_m3s": config.avg_discharge_m3s,
        "计算引擎": engine_label,
        "提示": "模型就绪。可调用 run_hydrodynamic_simulation 运行水动力模拟, "
                "或 simulate_pollution_event 模拟污染事件。",
    }, ensure_ascii=False)


# ---- Tool 2: 水动力模拟 ----

def tool_river_hydro_simulation(
    state: RiverModelState,
    upstream_flow_m3s: float = 6.9,
    downstream_stage_m: float = 0.83,
    tidal_amplitude_m: float = 0.8,
    tidal_period_h: float = 12.42,
    duration_h: float = 24.0,
    dt_s: float = 60.0,
) -> str:
    """运行雅瑶水道一维水动力模拟。"""
    if not state.model_initialized:
        return json.dumps({"error": "请先调用 init_river_model 初始化模型"}, ensure_ascii=False)

    try:
        result = run_hydrodynamic(
            state.config, state.cross_sections,
            upstream_flow_m3s, downstream_stage_m,
            tidal_amplitude_m, tidal_period_h,
            duration_h, dt_s, 30.0,
        )
    except Exception as e:
        return json.dumps({"error": f"水动力模拟失败: {str(e)}"}, ensure_ascii=False)

    state.hydro_result = result

    # 生成可视化
    try:
        plot_water_level_profile(result, state.config, "river_hydro_profile.png")
        plot_discharge_profile(result, "river_discharge_profile.png")
    except Exception:
        pass

    # 统计摘要
    wl_max = float(np.max(result.water_level))
    wl_min = float(np.min(result.water_level))
    v_max = float(np.max(result.velocity))
    v_mean = float(np.mean(np.abs(result.velocity)))
    q_max = float(np.max(np.abs(result.discharge)))

    return json.dumps({
        "状态": "水动力模拟完成",
        "模拟时长_h": duration_h,
        "时间步长_s": dt_s,
        "输出帧数": len(result.t),
        "最高水位_m": round(wl_max, 2),
        "最低水位_m": round(wl_min, 2),
        "水位变幅_m": round(wl_max - wl_min, 2),
        "最大流速_ms": round(v_max, 3),
        "平均流速_ms": round(v_mean, 3),
        "最大流量_m3s": round(q_max, 1),
        "引擎": result.params.get("engine", "unknown"),
        "输出文件": ["river_hydro_profile.png", "river_discharge_profile.png"],
        "提示": "水动力场就绪。可调用 simulate_pollution_event 模拟污染事件。",
    }, ensure_ascii=False)


# ---- Tool 3: 污染事件模拟 ----

def tool_river_pollution_event(
    state: RiverModelState,
    chainage_m: float = 1600.0,
    pollutant_type: str = "ammonia",
    load_kg: float = 50.0,
    duration_min: float = 30.0,
    simulation_hours: float = 24.0,
) -> str:
    """在雅瑶水道指定位置模拟污染事件。"""
    if not state.has_hydro():
        return json.dumps({
            "error": "请先运行 run_hydrodynamic_simulation",
            "hint": "水动力模拟提供流速场, 是水质模拟的前提",
        }, ensure_ascii=False)

    valid_types = ["ammonia", "cbod", "cod", "conservative"]
    if pollutant_type not in valid_types:
        return json.dumps({"error": f"无效污染物类型。可选: {valid_types}"}, ensure_ascii=False)

    try:
        result = solve_water_quality(
            state.config,
            state.hydro_result,
            pollutant_type=pollutant_type,
            load_kg=load_kg,
            chainage_m=chainage_m,
            duration_min=duration_min,
            simulation_hours=simulation_hours,
            dt_s=60.0,
        )
    except Exception as e:
        return json.dumps({"error": f"水质模拟失败: {str(e)}"}, ensure_ascii=False)

    state.wq_result = result
    scenario_name = f"{pollutant_type}_{load_kg}kg_{chainage_m}m"
    state.scenarios[scenario_name] = result.params

    # 生成可视化
    try:
        plot_concentration_contour(result, pollutant_type if pollutant_type != "cod" else "cbod",
                                    "river_wq_contour.png", state.config)
        plot_concentration_profile(result, simulation_hours * 0.5,
                                    pollutant_type if pollutant_type != "cod" else "cbod",
                                    "river_wq_profile.png", state.config)
        plot_concentration_timeseries(
            result,
            [0, result.ammonia.shape[1] // 4, result.ammonia.shape[1] // 2,
             3 * result.ammonia.shape[1] // 4, result.ammonia.shape[1] - 1],
            pollutant_type if pollutant_type != "cod" else "cbod",
            "river_wq_timeseries.png",
        )
        if state.has_hydro():
            plot_hydro_wq_dashboard(state.hydro_result, result, "river_dashboard.png", state.config)

        # 自动生成地图动画 (如果 GPS 数据可用)
        try:
            excel_default = (
                "苗圃杯/无人船智能水质监测分析和因果溯源样本数据/"
                "河道巡航走航样本数据.xlsx"
            )
            if os.path.exists(excel_default):
                from .geo_mapping import build_chainage_gps_mapping
                from .animation import build_pollution_map_animation
                gps_mapping = build_chainage_gps_mapping(excel_default, smoothing_sigma=3.0)
                build_pollution_map_animation(
                    gps_mapping, result,
                    pollutant_type if pollutant_type != "cod" else "cbod",
                    source_chainage_m=chainage_m,
                    output_path="river_pollution_animation.html",
                    max_frames=40,
                )
        except Exception:
            pass  # 动画失败不影响主流程
    except Exception:
        pass

    # 分析影响
    if pollutant_type == "ammonia" and result.ammonia is not None:
        data = result.ammonia
        std_limit = 2.0
    elif pollutant_type in ("cbod", "cod") and result.cbod is not None:
        data = result.cbod
        std_limit = 10.0
    else:
        data = result.ammonia if result.ammonia is not None else result.cbod
        std_limit = 2.0

    peak_c = float(np.max(data)) if data is not None else 0.0
    peak_time_idx = int(np.argmax(np.max(data, axis=1))) if data is not None else 0
    peak_time_h = float(result.t[peak_time_idx])

    # 增量影响范围 (浓度超过基线 1.5x 的断面数)
    if data is not None:
        baseline = data[0, :]
        # 找任意时刻浓度超过基线 1.5 倍的断面
        affected_mask = np.any(data > baseline * 1.5, axis=0)
        n_affected = int(np.sum(affected_mask))
        affected_chainage_range = (
            float(result.chainage[affected_mask][0]) if n_affected > 0 else 0,
            float(result.chainage[affected_mask][-1]) if n_affected > 0 else 0,
        )
    else:
        n_affected = 0
        affected_chainage_range = (0, 0)

    # 下游到达时间 (峰值到达最后一个断面)
    arrival_h = float(result.t[-1])
    if data is not None:
        last_cs = data[:, -1]
        exceed_factor = 1.3
        exceeded = np.where(last_cs > last_cs[0] * exceed_factor)[0]
        arrival_h = float(result.t[exceeded[0]]) if len(exceeded) > 0 else float(result.t[-1])

    return json.dumps({
        "状态": "污染事件模拟完成",
        "污染物": pollutant_type,
        "排放量_kg": load_kg,
        "排放位置_m": chainage_m,
        "排放时长_min": duration_min,
        "模拟时长_h": simulation_hours,
        "峰值浓度_mgL": round(peak_c, 2),
        "峰值时刻_h": round(peak_time_h, 1),
        "超标倍数": round(peak_c / std_limit, 1) if std_limit > 0 else 0,
        "显著影响断面数": n_affected,
        "影响桩号范围_m": list(affected_chainage_range),
        "下游到达时间_h": round(arrival_h, 1),
        "情景名称": scenario_name,
        "输出文件": ["river_wq_contour.png", "river_wq_profile.png",
                   "river_wq_timeseries.png", "river_dashboard.png",
                   "river_pollution_animation.html"],
        "提示": f"峰值浓度 {peak_c:.1f} mg/L 为标准({std_limit} mg/L)的 {peak_c/std_limit:.1f} 倍。"
                f"可调用 get_concentration_profile 查询详细剖面。",
    }, ensure_ascii=False)


# ---- Tool 4: 浓度剖面查询 ----

def tool_river_concentration_profile(
    state: RiverModelState,
    time_h: float = 6.0,
    constituent: str = "ammonia",
) -> str:
    """查询指定时刻的沿程浓度剖面。"""
    if not state.has_wq():
        return json.dumps({"error": "请先运行 simulate_pollution_event"}, ensure_ascii=False)

    wq = state.wq_result
    if constituent == "ammonia":
        data = wq.ammonia
        std = 2.0
    elif constituent in ("cbod", "cod"):
        data = wq.cbod
        std = 10.0
    elif constituent == "do":
        data = wq.dissolved_oxygen
        std = 2.0
    else:
        return json.dumps({"error": f"未知组分: {constituent}"}, ensure_ascii=False)

    if data is None:
        return json.dumps({"error": f"组分 {constituent} 数据不可用"}, ensure_ascii=False)

    t_idx = int(np.argmin(np.abs(wq.t - time_h)))
    t_idx = min(t_idx, len(wq.t) - 1)
    profile = data[t_idx, :]
    chainage = wq.chainage

    # 超标断面
    exceeded_mask = profile > std
    exceeded_sections = [
        {"桩号_m": round(float(chainage[i]), 0), "浓度": round(float(profile[i]), 2)}
        for i in range(len(profile)) if exceeded_mask[i]
    ]

    return json.dumps({
        "时刻_h": round(float(wq.t[t_idx]), 1),
        "组分": constituent,
        "标准限值_mgL": std,
        "沿程浓度": [round(float(c), 2) for c in profile[::5]],  # 抽样
        "桩号_m": [round(float(ch), 0) for ch in chainage[::5]],
        "最大浓度_mgL": round(float(np.max(profile)), 2),
        "最大浓度位置_m": round(float(chainage[int(np.argmax(profile))]), 0),
        "超标断面数": int(np.sum(exceeded_mask)),
        "超标断面": exceeded_sections[:8],  # 最多8个
        "影响范围_m": round(float(
            chainage[exceeded_mask][-1] - chainage[exceeded_mask][0]
        ) if np.sum(exceeded_mask) >= 1 else 0, 0),
    }, ensure_ascii=False)


# ---- Tool 5: 情景对比 ----

def tool_river_compare_scenarios(
    state: RiverModelState,
    scenario1_name: str = "",
    scenario2_name: str = "",
) -> str:
    """对比两个已保存的模拟情景。"""
    if not state.scenarios:
        return json.dumps({"error": "没有已保存的情景。请先运行 simulate_pollution_event"}, ensure_ascii=False)

    names = list(state.scenarios.keys())
    if not scenario1_name and len(names) >= 1:
        scenario1_name = names[-1] if len(names) >= 2 else names[0]
    if not scenario2_name and len(names) >= 2:
        scenario2_name = names[-2]
    elif not scenario2_name:
        return json.dumps({
            "error": "需要至少两个情景才能对比",
            "可用情景": names,
        }, ensure_ascii=False)

    s1 = state.scenarios.get(scenario1_name, {})
    s2 = state.scenarios.get(scenario2_name, {})

    return json.dumps({
        "情景1": {"名称": scenario1_name, **s1},
        "情景2": {"名称": scenario2_name, **s2},
        "可用情景列表": names,
    }, ensure_ascii=False)


# ---- Tool 6: 数据导出 ----

def tool_river_export_data(
    state: RiverModelState,
    format: str = "json",
    scenario_name: str = "latest",
) -> str:
    """导出模拟结果数据。"""
    wq = state.wq_result
    if wq is None:
        return json.dumps({"error": "没有可用的模拟结果"}, ensure_ascii=False)

    if format == "json":
        # 导出为精简 JSON (浓度沿程剖面, 所有时刻)
        export = {
            "chainage_m": wq.chainage.tolist(),
            "time_h": wq.t.tolist(),
        }
        if wq.ammonia is not None:
            export["ammonia_mgL"] = wq.ammonia.tolist()
        if wq.cbod is not None:
            export["cbod_mgL"] = wq.cbod.tolist()
        if wq.dissolved_oxygen is not None:
            export["do_mgL"] = wq.dissolved_oxygen.tolist()

        out_path = "river_model_export.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)
        return json.dumps({
            "状态": "数据已导出",
            "文件": out_path,
            "格式": "JSON",
            "大小_字节": os.path.getsize(out_path),
        }, ensure_ascii=False)

    elif format == "csv":
        import pandas as pd
        out_path = "river_model_export.csv"
        rows = []
        for ti in range(len(wq.t)):
            for ci in range(len(wq.chainage)):
                row = {"time_h": wq.t[ti], "chainage_m": wq.chainage[ci]}
                if wq.cbod is not None:
                    row["cbod_mgL"] = wq.cbod[ti, ci]
                if wq.ammonia is not None:
                    row["ammonia_mgL"] = wq.ammonia[ti, ci]
                if wq.dissolved_oxygen is not None:
                    row["do_mgL"] = wq.dissolved_oxygen[ti, ci]
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return json.dumps({
            "状态": "数据已导出",
            "文件": out_path,
            "格式": "CSV",
            "行数": len(df),
        }, ensure_ascii=False)

    else:
        return json.dumps({"error": f"不支持的格式: {format}, 可选 'json' 或 'csv'"}, ensure_ascii=False)


# ---- Tool 7: 地图污染扩散动画 ----

def tool_river_map_animation(
    state: RiverModelState,
    excel_path: str = "苗圃杯/无人船智能水质监测分析和因果溯源样本数据/河道巡航走航样本数据.xlsx",
    constituent: str = "ammonia",
    output_path: str = "river_pollution_animation.html",
    source_chainage_m: float | None = None,
    dual_view: bool = True,
    max_frames: int = 60,
) -> str:
    """生成污染扩散地图动画 HTML。

    基于走航船 GPS 轨迹和 ADR 污染模拟结果，
    生成带时间轴的 Folium 地图动画和 Plotly 浓度剖面图。
    """
    if not state.has_wq():
        return json.dumps({
            "error": "请先运行 simulate_pollution_event",
            "hint": "需要污染模拟结果作为动画数据源",
        }, ensure_ascii=False)

    # 推断污染源桩号
    wq = state.wq_result
    if source_chainage_m is None and wq.params:
        source_chainage_m = wq.params.get("chainage_m", 1600.0)

    try:
        from .geo_mapping import build_chainage_gps_mapping
        from .animation import build_pollution_map_animation, build_dual_view_animation

        # 第一步: 建立 GPS 映射
        gps_mapping = build_chainage_gps_mapping(excel_path, smoothing_sigma=3.0)

        # 第二步: 生成动画
        if dual_view:
            result_path = build_dual_view_animation(
                gps_mapping, wq, constituent,
                source_chainage_m=source_chainage_m,
                output_path=output_path,
                max_frames=max_frames,
            )
        else:
            result_path = build_pollution_map_animation(
                gps_mapping, wq, constituent,
                source_chainage_m=source_chainage_m,
                output_path=output_path,
                max_frames=max_frames,
            )

        file_size_mb = round(os.path.getsize(result_path) / 1048576, 2)
        return json.dumps({
            "状态": "动画已生成",
            "HTML文件": result_path,
            "大小_MB": file_size_mb,
            "组分": constituent,
            "时间帧数": min(max_frames, len(wq.t)),
            "时间范围_h": f"{wq.t[0]:.1f} ~ {wq.t[-1]:.1f}",
            "河道长度_km": round(gps_mapping["total_length_m"] / 1000, 2),
            "GPS点数": gps_mapping["point_count"],
            "污染源桩号_m": source_chainage_m,
            "标准": "GB3838-2002",
            "播放": "用浏览器打开 HTML 文件，点击播放按钮即可观看",
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"动画生成失败: {str(e)}"}, ensure_ascii=False)
