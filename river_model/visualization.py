"""河道模型可视化: 纵横剖面、浓度过程线、仪表盘图。"""

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import YayaoConfig
from .state import HydroResult, WQResult

# 中文字体
try:
    from water_quality_mcp.src.water_quality_mcp.font_utils import discover_font
    _font_prop = discover_font()
except Exception:
    _font_prop = None

def _fp():
    return _font_prop


# ---- 水动力可视化 ----

def plot_water_level_profile(hydro: HydroResult, config: YayaoConfig,
                              output_path: str = "river_hydro_profile.png",
                              time_indices: Optional[list] = None):
    """纵剖面水位线 (指定时刻)。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    if time_indices is None:
        n = hydro.water_level.shape[0]
        time_indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(time_indices)))
    for idx, c in zip(time_indices, colors):
        idx = min(idx, hydro.water_level.shape[0] - 1)
        ax.plot(hydro.chainage, hydro.water_level[idx, :],
                color=c, linewidth=1.5, label=f"t = {hydro.t[idx]:.1f} h")
    ax.set_xlabel("桩号 (m)", fontproperties=_fp())
    ax.set_ylabel("水位 (m, 1985高程)", fontproperties=_fp())
    ax.set_title("雅瑶水道 — 沿程水位剖面", fontproperties=_fp(), fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_discharge_profile(hydro: HydroResult, output_path: str = "river_discharge_profile.png",
                            time_indices: Optional[list] = None):
    """流量纵剖面。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    if time_indices is None:
        n = hydro.discharge.shape[0]
        time_indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(time_indices)))
    for idx, c in zip(time_indices, colors):
        idx = min(idx, hydro.discharge.shape[0] - 1)
        ax.plot(hydro.chainage, hydro.discharge[idx, :],
                color=c, linewidth=1.5, label=f"t = {hydro.t[idx]:.1f} h")
    ax.set_xlabel("桩号 (m)", fontproperties=_fp())
    ax.set_ylabel("流量 (m³/s)", fontproperties=_fp())
    ax.set_title("雅瑶水道 — 沿程流量剖面", fontproperties=_fp(), fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_velocity_timeseries(hydro: HydroResult, section_ids: list,
                              output_path: str = "river_velocity_ts.png"):
    """指定断面流速过程线。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    for sid in section_ids:
        sid = min(sid, hydro.velocity.shape[1] - 1)
        ch = hydro.chainage[sid]
        ax.plot(hydro.t, hydro.velocity[:, sid], linewidth=1.2,
                label=f"桩号 {ch:.0f}m")
    ax.set_xlabel("时间 (h)", fontproperties=_fp())
    ax.set_ylabel("流速 (m/s)", fontproperties=_fp())
    ax.set_title("雅瑶水道 — 流速过程线", fontproperties=_fp(), fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---- 水质可视化 ----

def plot_concentration_contour(wq: WQResult, constituent: str = "ammonia",
                                output_path: str = "river_wq_contour.png",
                                config: Optional[YayaoConfig] = None):
    """浓度 x-t 等值线图。"""
    if constituent == "ammonia":
        data = wq.ammonia
        label = "氨氮 (mg/L)"
        vline_val = 2.0
    elif constituent == "cbod":
        data = wq.cbod
        label = "CBOD (mg/L)"
        vline_val = 10.0
    elif constituent == "do":
        data = wq.dissolved_oxygen
        label = "溶解氧 (mg/L)"
        vline_val = 2.0
    else:
        return

    if data is None:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    X, Y = np.meshgrid(wq.chainage, wq.t)
    levels = np.linspace(np.nanmin(data), np.nanmax(data), 40)
    cf = ax.contourf(X, Y, data, levels=levels, cmap="YlOrRd", extend="max")
    cbar = fig.colorbar(cf, ax=ax, label=label)
    ax.axhline(y=wq.t[-1] * 0.1, color='blue', linestyle='--', alpha=0.5)  # 源位置标记

    ax.set_xlabel("桩号 (m)", fontproperties=_fp())
    ax.set_ylabel("时间 (h)", fontproperties=_fp())
    ax.set_title(f"雅瑶水道 — {label} 时空分布", fontproperties=_fp(), fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_concentration_profile(wq: WQResult, time_h: float,
                                constituent: str = "ammonia",
                                output_path: str = "river_wq_profile.png",
                                config: Optional[YayaoConfig] = None):
    """指定时刻的沿程浓度剖面。"""
    if constituent == "ammonia":
        data = wq.ammonia
        std_val = 2.0
        ylabel = "氨氮 (mg/L)"
    elif constituent == "cbod":
        data = wq.cbod
        std_val = 10.0
        ylabel = "CBOD (mg/L)"
    elif constituent == "do":
        data = wq.dissolved_oxygen
        std_val = 2.0
        ylabel = "溶解氧 (mg/L)"
    else:
        return

    if data is None:
        return

    t_idx = int(np.argmin(np.abs(wq.t - time_h)))
    t_idx = min(t_idx, len(wq.t) - 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wq.chainage, data[t_idx, :], 'b-', linewidth=1.5)
    ax.axhline(y=std_val, color='r', linestyle='--', alpha=0.7,
               label=f"V类标准 ({std_val} mg/L)")
    # 超标区着色
    above = data[t_idx, :] > std_val
    if above.any():
        ax.fill_between(wq.chainage, std_val, data[t_idx, :],
                         where=above, color='red', alpha=0.15)

    ax.set_xlabel("桩号 (m)", fontproperties=_fp())
    ax.set_ylabel(ylabel, fontproperties=_fp())
    ax.set_title(f"雅瑶水道 — t={wq.t[t_idx]:.1f}h {ylabel}沿程剖面", fontproperties=_fp(), fontsize=13)
    ax.legend(fontsize=9, prop=_fp())
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_concentration_timeseries(wq: WQResult, section_ids: list,
                                   constituent: str = "ammonia",
                                   output_path: str = "river_wq_timeseries.png"):
    """选定断面的浓度过程线。"""
    if constituent == "ammonia":
        data = wq.ammonia
        std_val = 2.0
        ylabel = "氨氮 (mg/L)"
    elif constituent == "cbod":
        data = wq.cbod
        std_val = 10.0
        ylabel = "CBOD (mg/L)"
    elif constituent == "do":
        data = wq.dissolved_oxygen
        std_val = 2.0
        ylabel = "溶解氧 (mg/L)"
    else:
        return

    if data is None:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for sid in section_ids:
        sid = min(sid, data.shape[1] - 1)
        ch = wq.chainage[sid]
        ax.plot(wq.t, data[:, sid], linewidth=1.2, label=f"桩号 {ch:.0f}m")
    ax.axhline(y=std_val, color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel("时间 (h)", fontproperties=_fp())
    ax.set_ylabel(ylabel, fontproperties=_fp())
    ax.set_title(f"雅瑶水道 — {ylabel} 浓度过程线", fontproperties=_fp(), fontsize=13)
    ax.legend(fontsize=8, prop=_fp())
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_hydro_wq_dashboard(hydro: HydroResult, wq: WQResult,
                             output_path: str = "river_dashboard.png",
                             config: Optional[YayaoConfig] = None):
    """四面板仪表盘: 水位剖面 + 浓度等值线 + DO剖面 + 氨氮过程线。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # (0,0) 水位剖面 (首/中/末三时刻)
    n = hydro.water_level.shape[0]
    for idx in [0, n // 2, n - 1]:
        axes[0, 0].plot(hydro.chainage, hydro.water_level[idx, :],
                        linewidth=1.2, label=f"t={hydro.t[idx]:.1f}h")
    axes[0, 0].set_xlabel("桩号 (m)", fontproperties=_fp())
    axes[0, 0].set_ylabel("水位 (m)", fontproperties=_fp())
    axes[0, 0].set_title("沿程水位", fontproperties=_fp())
    axes[0, 0].legend(fontsize=7, prop=_fp())
    axes[0, 0].grid(True, alpha=0.3)

    # (0,1) 浓度等值线
    if wq.ammonia is not None:
        X, Y = np.meshgrid(wq.chainage, wq.t)
        levels = np.linspace(np.nanmin(wq.ammonia), np.nanmax(wq.ammonia), 30)
        cf = axes[0, 1].contourf(X, Y, wq.ammonia, levels=levels,
                                  cmap="YlOrRd", extend="max")
        fig.colorbar(cf, ax=axes[0, 1], label="氨氮 (mg/L)")
    axes[0, 1].set_xlabel("桩号 (m)", fontproperties=_fp())
    axes[0, 1].set_ylabel("时间 (h)", fontproperties=_fp())
    axes[0, 1].set_title("氨氮时空分布", fontproperties=_fp())

    # (1,0) DO剖面 (首/中/末)
    if wq.dissolved_oxygen is not None:
        n_wq = wq.dissolved_oxygen.shape[0]
        for idx in [0, n_wq // 2, n_wq - 1]:
            axes[1, 0].plot(wq.chainage, wq.dissolved_oxygen[idx, :],
                            linewidth=1.2, label=f"t={wq.t[idx]:.1f}h")
    axes[1, 0].axhline(y=2.0, color='r', linestyle='--', linewidth=0.8)
    axes[1, 0].set_xlabel("桩号 (m)", fontproperties=_fp())
    axes[1, 0].set_ylabel("溶解氧 (mg/L)", fontproperties=_fp())
    axes[1, 0].set_title("沿程溶解氧", fontproperties=_fp())
    axes[1, 0].legend(fontsize=7, prop=_fp())
    axes[1, 0].grid(True, alpha=0.3)

    # (1,1) 氨氮过程线 (上游/源/下游)
    if wq.ammonia is not None:
        ncs = wq.ammonia.shape[1]
        key_ids = [0, ncs // 4, ncs // 2, 3 * ncs // 4, ncs - 1]
        for sid in key_ids:
            axes[1, 1].plot(wq.t, wq.ammonia[:, sid], linewidth=1.2,
                            label=f"桩号 {wq.chainage[sid]:.0f}m")
    axes[1, 1].axhline(y=2.0, color='r', linestyle='--', linewidth=0.8,
                       label="V类标准 (2.0 mg/L)")
    axes[1, 1].set_xlabel("时间 (h)", fontproperties=_fp())
    axes[1, 1].set_ylabel("氨氮 (mg/L)", fontproperties=_fp())
    axes[1, 1].set_title("氨氮浓度过程线", fontproperties=_fp())
    axes[1, 1].legend(fontsize=7, prop=_fp())
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("雅瑶水道 水动力-水质模拟仪表盘",
                 fontproperties=_fp(), fontsize=15, y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================
#  Plotly 交互式浓度剖面动画
# ============================================================

def build_plotly_animation(
    wq_result,
    constituent: str = "ammonia",
    source_chainage_m: float | None = None,
):
    """生成 Plotly 交互式浓度剖面动画。

    Args:
        wq_result: WQResult 对象
        constituent: ammonia / cbod / do
        source_chainage_m: 污染源桩号, None 则从 wq_result.params 推断

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    data_map = {"ammonia": wq_result.ammonia, "cbod": wq_result.cbod,
                "do": wq_result.dissolved_oxygen}
    if constituent not in data_map or data_map[constituent] is None:
        fig = go.Figure()
        fig.update_layout(title=f"无 {constituent} 数据")
        return fig

    data = data_map[constituent]
    chainage = wq_result.chainage
    t_hours = wq_result.t

    standards = {
        "ammonia": {"name": "氨氮", "unit": "mg/L", "limit": 2.0,
                     "gb_levels": [0.15, 0.5, 1.0, 1.5, 2.0],
                     "gb_colors": ["#0066FF", "#00CC44", "#FFCC00", "#FF8800", "#FF0000"],
                     "gb_names": ["I类", "II类", "III类", "IV类", "V类"]},
        "cbod": {"name": "CBOD", "unit": "mg/L", "limit": 10.0,
                  "gb_levels": [3, 4, 6, 10, 15],
                  "gb_colors": ["#0066FF", "#00CC44", "#FFCC00", "#FF8800", "#FF0000"],
                  "gb_names": ["I类", "II类", "III类", "IV类", "V类"]},
        "do": {"name": "溶解氧", "unit": "mg/L", "limit": 2.0, "inverted": True,
                "gb_levels": [7.5, 6.0, 5.0, 3.0, 2.0],
                "gb_colors": ["#0066FF", "#00CC44", "#FFCC00", "#FF8800", "#FF0000"],
                "gb_names": ["I类", "II类", "III类", "IV类", "V类"]},
    }
    std = standards.get(constituent, standards["ammonia"])

    if source_chainage_m is None and wq_result.params:
        source_chainage_m = wq_result.params.get("chainage_m", None)

    # 降采样
    nt = len(t_hours)
    max_f = 60
    if nt <= max_f:
        frame_indices = list(range(nt))
    else:
        frame_indices = np.linspace(0, nt - 1, max_f, dtype=int).tolist()

    # GB3838 色带
    traces = []
    for i, level in enumerate(std["gb_levels"]):
        c = std["gb_colors"][i]
        rgb = f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.12)"
        y_prev = 0 if i == 0 else std["gb_levels"][i - 1]
        traces.append(go.Scatter(
            x=[chainage[0], chainage[-1], chainage[-1], chainage[0], chainage[0]],
            y=[y_prev, y_prev, level, level, y_prev],
            fill="toself", fillcolor=rgb, line={"width": 0},
            name=f"GB {std['gb_names'][i]}", showlegend=True, hoverinfo="skip",
        ))
    last = std["gb_levels"][-1]
    traces.append(go.Scatter(
        x=[chainage[0], chainage[-1], chainage[-1], chainage[0], chainage[0]],
        y=[last, last, last * 5, last * 5, last],
        fill="toself", fillcolor="rgba(153,0,0,0.12)", line={"width": 0},
        name="劣V类", showlegend=True, hoverinfo="skip",
    ))

    # 标准限值线
    traces.append(go.Scatter(
        x=[chainage[0], chainage[-1]], y=[std["limit"], std["limit"]],
        mode="lines", line={"color": "red", "width": 2, "dash": "dash"},
        name=f"V类标准 ({std['limit']})", hoverinfo="skip",
    ))

    # 污染源标记
    if source_chainage_m is not None:
        traces.append(go.Scatter(
            x=[source_chainage_m, source_chainage_m],
            y=[0, float(np.max(data)) * 1.1],
            mode="lines", line={"color": "orange", "width": 2, "dash": "dot"},
            name=f"污染源 {source_chainage_m:.0f}m", hoverinfo="skip",
        ))

    # 浓度曲线 (初始帧)
    traces.append(go.Scatter(
        x=list(chainage), y=list(data[frame_indices[0], :]),
        mode="lines", line={"color": "#00D4FF", "width": 2.5},
        name="浓度", fill="tozeroy", fillcolor="rgba(0,212,255,0.15)",
    ))

    # Frames
    frames = []
    for fi in frame_indices:
        frames.append(go.Frame(
            data=[go.Scatter(
                x=list(chainage), y=list(data[fi, :]),
                mode="lines", line={"color": "#00D4FF", "width": 2.5},
                name="浓度", fill="tozeroy", fillcolor="rgba(0,212,255,0.15)",
            )],
            name=f"f{fi}",
            layout={"annotations": [go.layout.Annotation(
                text=f"t = {t_hours[fi]:.1f} h",
                x=0.98, y=0.95, xref="paper", yref="paper",
                showarrow=False, font={"size": 16, "color": "#FFCC00"},
                bgcolor="rgba(0,0,0,0.5)", borderpad=6,
            )]},
        ))

    # Slider
    steps = []
    for i, fi in enumerate(frame_indices):
        steps.append({
            "args": [[f"f{fi}"], {"frame": {"duration": 100, "redraw": True},
                                    "mode": "immediate", "fromcurrent": True,
                                    "transition": {"duration": 80}}],
            "label": f"{t_hours[fi]:.1f}h",
            "method": "animate",
        })
    sliders = [{
        "active": 0, "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 16, "color": "#FFCC00"},
                          "prefix": "时间: ", "suffix": " h",
                          "visible": True, "xanchor": "right"},
        "transition": {"duration": 80}, "pad": {"b": 10, "t": 50},
        "len": 0.9, "x": 0.1, "y": 0, "steps": steps,
    }]

    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        title={"text": f"雅瑶水道 {std['name']} 污染扩散剖面<br>"
                        f"<sub>泄漏量 {wq_result.params.get('load_kg','?')} kg | "
                        f"模拟 {t_hours[-1]:.0f} h</sub>",
               "x": 0.5, "xanchor": "center", "font": {"size": 18}},
        xaxis={"title": "桩号 (m)", "range": [0, chainage[-1]], "gridcolor": "#333"},
        yaxis={"title": f"{std['name']} ({std['unit']})",
               "range": [0, max(float(np.max(data)) * 1.1, std["limit"] * 3)],
               "gridcolor": "#333"},
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        font={"color": "#CCCCCC", "size": 13},
        sliders=sliders,
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "▶ 播放", "method": "animate",
                 "args": [None, {"frame": {"duration": 100, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 80}}]},
                {"label": "⏸ 暂停", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate", "transition": {"duration": 0}}]},
            ],
            "direction": "left", "pad": {"r": 10, "t": 70},
            "showactive": True, "x": 0.1, "y": 0,
            "xanchor": "right", "yanchor": "top",
        }],
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "right", "x": 1, "font": {"size": 10}},
        margin={"l": 60, "r": 30, "t": 90, "b": 60},
    )
    return fig
