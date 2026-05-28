r"""HEC-RAS 桥接模块 — 生成输入文件 + COM 驱动计算 + 提取结果。

HEC-RAS 7.0 安装于 D:\my_HEC-RAS。COM 需管理员运行注册脚本后可用。
接口: win32com.client.Dispatch("RAS70.HECRASController")
几何文件格式精确匹配 HEC-RAS 7.0 原生输出。
"""

import os
import time
import tempfile
from typing import List, Optional

import numpy as np

from .config import YayaoConfig
from .cross_sections import CrossSection
from .state import HydroResult


def _get_hecras_com():
    """获取 HEC-RAS COM 控制器实例。"""
    import win32com.client
    return win32com.client.Dispatch("RAS70.HECRASController")


def is_hecras_available() -> bool:
    path = r"D:\my_HEC-RAS\Ras.exe"
    return os.path.exists(path)


# ---- HEC-RAS 输入文件生成 ----

_PRJ_GEOM_ONLY = """Proj Title={title}
Current Geom=g01
Default Exp/Contr=0.3,0.1
SI Units
Geom File=g01
Y Axis Title=Elevation
X Axis Title(PF)=Main Channel Distance
X Axis Title(XS)=Station
BEGIN DESCRIPTION:

END DESCRIPTION:
DSS Start Date=
DSS Start Time=
DSS End Date=
DSS End Time=
DSS Export Filename=
DSS Export Rating Curves= 0
DSS Export Rating Curve Sorted= 0
DSS Export Volume Flow Curves= 0
DXF Filename=
DXF OffsetX= 0
DXF OffsetY= 0
DXF ScaleX= 1
DXF ScaleY= 10
GIS Export Profiles= 0
"""

_PRJ_FULL = """Proj Title={title}
Current Plan=p01
Current Geom=g01
Current Flow=f01
Default Exp/Contr=0.3,0.1
SI Units
Geom File=g01
Unsteady Flow File=u01
Plan File=p01
Flow File=f01
Y Axis Title=Elevation
X Axis Title(PF)=Main Channel Distance
X Axis Title(XS)=Station
BEGIN DESCRIPTION:

END DESCRIPTION:
DSS Start Date=
DSS Start Time=
DSS End Date=
DSS End Time=
DSS Export Filename=
DSS Export Rating Curves= 0
DSS Export Rating Curve Sorted= 0
DSS Export Volume Flow Curves= 0
DXF Filename=
DXF OffsetX= 0
DXF OffsetY= 0
DXF ScaleX= 1
DXF ScaleY= 10
GIS Export Profiles= 0
"""


def _build_project_file(work_dir: str, title: str = "Yayao",
                        include_flow_plan: bool = False) -> str:
    path = os.path.join(work_dir, "Yayao.prj")
    tmpl = _PRJ_FULL if include_flow_plan else _PRJ_GEOM_ONLY
    with open(path, "w", encoding="utf-8") as f:
        f.write(tmpl.format(title=title))
    return path


def _build_geometry_file(work_dir: str, config: YayaoConfig,
                         cross_sections: List[CrossSection]) -> str:
    """生成 HEC-RAS 7.0 几何文件 (.g01) — 精确匹配 HEC-RAS 7.0 原生格式。"""
    n = len(cross_sections)

    lines = [
        "Geom Title=Yayao",
        "Program Version=7.00",
        "Viewing Rectangle= 0 , 1 , 1 , 0 ",
        "",
        "River Reach=Yayao             ,Main            ",
        "Reach XY= 2 ",
        "-0.20228215767630.726141078838170.567427385892120.56224066390042",
        "Rch Text X Y=-0.0098548,0.685166",
        "Reverse River Text= 0 ",
        "",
    ]

    for i, cs in enumerate(cross_sections):
        # HEC-RAS River Station = 上游大, 下游小
        station = int(config.length_m - cs.chainage_m)
        station = max(station, 0)

        if i < n - 1:
            dx = int(round(cross_sections[i + 1].chainage_m - cs.chainage_m))
        else:
            dx = 50

        # Type RM line — EXACT HEC-RAS 7.0 format
        lines.append(f"Type RM Length L Ch R = 1 ,{station:<8d},{dx},{dx},{dx}")
        lines.append(f"Node Last Edited Time=May/27/2026 00:00:00")

        s = cs.stations
        n_pts = len(s)
        lines.append(f"#Sta/Elev= {n_pts} ")

        # Sta/Elev: 8-char fixed-width, sta without decimals, el with 1 decimal
        vals = ""
        for j in range(n_pts):
            vals += f"{float(s[j, 0]):8.0f}{float(s[j, 1]):8.1f}"
        lines.append(vals)

        # Manning n — 3 triplets of (station, n_value, 0), n without leading zero
        lines.append("#Mann= 3 ,0,0")
        left_edge = int(round(float(s[0, 0])))
        left_bank = int(round(float(s[1, 0]))) if n_pts >= 4 else left_edge + 5
        right_bank = int(round(float(s[-2, 0]))) if n_pts >= 4 else -left_edge - 5
        def _fmt_n(v):
            s = f"{v:.3f}".lstrip('0')
            return f"{s:>8}"
        lines.append(
            f"{left_edge:8d}" + _fmt_n(cs.mannings_n_left) + f"{0:8d}"
            + f"{left_bank:8d}" + _fmt_n(cs.mannings_n_channel) + f"{0:8d}"
            + f"{right_bank:8d}" + _fmt_n(cs.mannings_n_right) + f"{0:8d}"
        )

        lines.append(f"Bank Sta={left_bank},{right_bank}")
        lines.append("XS Rating Curve= 0 ,0")
        bed_el = float(s[1, 1])
        lines.append(f"XS HTab Starting El and Incr={bed_el + 0.5:.1f},0.15, 20 ")
        lines.append("XS HTab Horizontal Distribution= 5 , 5 , 5 ")
        lines.append("Exp/Cntr=0.3,0.1")
        lines.append("")

    # Footer — exact HEC-RAS 7.0 format
    lines.append("LCMann Time=Dec/30/1899 00:00:00")
    lines.append("LCMann Region Time=Dec/30/1899 00:00:00")
    lines.append("LCMann Table=0")
    lines.append("Chan Stop Cuts=-1 ")
    lines.append("")
    lines.append("")
    lines.append("Use User Specified Reach Order=0")
    lines.append("GIS Ratio Cuts To Invert=-1")
    lines.append("GIS Limit At Bridges=0")
    lines.append("Composite Channel Slope=5")
    lines.append("")

    # 文件名必须是 <ProjectName>.g01 以匹配 PRJ 中的 Geom File=g01
    path = os.path.join(work_dir, "Yayao.g01")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def _build_unsteady_flow_file(work_dir: str, config: YayaoConfig,
                               upstream_flow_m3s: float, downstream_stage_m: float,
                               tidal_amplitude_m: float, tidal_period_h: float,
                               duration_h: float, dt_s: float) -> str:
    """生成非恒定流边界条件文件 (.u01)。"""
    T_tide = tidal_period_h * 3600.0
    dt_min = dt_s / 60.0
    total_minutes = int(duration_h * 60)
    n_steps = int(total_minutes / dt_min) + 1

    lines = [
        "Flow Title=Yayao",
        "Program Version=7.00",
        "",
        "Boundary Location=3200.000",
        "Flow Hydrograph=1",
        "Flow Hydrograph Name=Upstream_Q",
        f"Flow Hydrograph Data={n_steps}",
        f"Flow Hydrograph Time Step={int(dt_min)}min",
        "Flow Hydrograph Date=01Jan2024",
    ]
    for m in range(n_steps):
        t_sec = m * dt_s
        q = upstream_flow_m3s * (1.0 + 0.05 * np.sin(2 * np.pi * t_sec / T_tide))
        lines.append(f"Flow={q:.4f}")

    lines += [
        "",
        "Boundary Location=0.000",
        "Stage Hydrograph=1",
        "Stage Hydrograph Name=Downstream_H",
        f"Stage Hydrograph Data={n_steps}",
        f"Stage Hydrograph Time Step={int(dt_min)}min",
        "Stage Hydrograph Date=01Jan2024",
    ]
    for m in range(n_steps):
        t_sec = m * dt_s
        stage = downstream_stage_m + tidal_amplitude_m * np.sin(2 * np.pi * t_sec / T_tide)
        lines.append(f"Stage={stage:.4f}")

    lines += [
        "",
        f"Initial Flow={upstream_flow_m3s:.4f}",
        f"Initial Stage={downstream_stage_m:.4f}",
    ]

    path = os.path.join(work_dir, "Yayao.u01")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def _build_plan_file(work_dir: str, duration_h: float, dt_s: float,
                     output_interval_min: float) -> str:
    """生成计算方案文件 (.p01)。"""
    end_h = int(duration_h)
    end_m = int((duration_h - end_h) * 60)
    lines = [
        "Plan Title=Yayao Simulation",
        "Program Version=7.00",
        "",
        "Geom File=g01",
        "Unsteady Flow File=u01",
        "Flow File=f01",
        "",
        "Simulation Date=01Jan2024",
        "Simulation Time=00:00:00",
        "Simulation End Date=01Jan2024",
        f"Simulation End Time={end_h:02d}:{end_m:02d}:00",
        f"Computation Interval={dt_s:.1f}sec",
        "Computation Level Option=1",
        f"Output Interval={output_interval_min:.1f}min",
        "Mixed Flow Regime=0",
        "Unsteady Flow Computational Scheme=1",
    ]
    path = os.path.join(work_dir, "Yayao.p01")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def _build_f01_file(work_dir: str, upstream_flow_m3s: float,
                    downstream_stage_m: float) -> str:
    """生成稳态流数据文件 (.f01) — 精确匹配 HEC-RAS 7.0 格式。"""
    content = f"Flow Title=Flow 01\nProgram Version=7.00\n"
    content += f"Number of Profiles= 1 \nProfile Names=PF 1\n"
    content += f"River Rch & RM=Yayao             ,Main            ,3200    \n     {upstream_flow_m3s}\n"
    content += f"Boundary for River Rch & Prof#=Yayao             ,Main            , 1 \n"
    content += f"Up Type= 0 \nDn Type= 1 \nDn Known WS={downstream_stage_m}\n"
    content += f"DSS Import StartDate=\nDSS Import StartTime=\n"
    content += f"DSS Import EndDate=\nDSS Import EndTime=\n"
    content += f"DSS Import GetInterval= 0 \nDSS Import Interval=\n"
    content += f"DSS Import GetPeak= 0 \nDSS Import FillOption= 0 \n"
    path = os.path.join(work_dir, "Yayao.f01")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ---- COM 驱动的水动力模拟 ----

def run_hecras_simulation(
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
    """通过 HEC-RAS COM 运行水动力模拟。

    当前 HEC-RAS 7.0 非恒定流 plan 格式未完全兼容 — 只跑稳态流初始化。
    非恒定流结果由 hydro_engine.run_hydrodynamic() 降级到纯 Python 求解。
    此函数返回稳态流结果，或抛出异常由上层捕获后降级。
    """
    if not is_hecras_available():
        raise RuntimeError("HEC-RAS 不可用: Ras.exe 未找到")

    work_dir = tempfile.mkdtemp(prefix="hecras_yayao_")

    try:
        # 1. 生成所有输入文件
        prj_path = _build_project_file(work_dir, include_flow_plan=True)
        _build_geometry_file(work_dir, config, cross_sections)
        _build_unsteady_flow_file(work_dir, config,
                                   upstream_flow_m3s, downstream_stage_m,
                                   tidal_amplitude_m, tidal_period_h,
                                   duration_h, dt_s)
        _build_plan_file(work_dir, duration_h, dt_s, output_interval_min)
        _build_f01_file(work_dir, upstream_flow_m3s, downstream_stage_m)

        # 2. COM 打开项目
        ras = _get_hecras_com()
        ras.Project_Open(prj_path)
        time.sleep(1.0)

        n_xs = ras.Schematic_XSCount()
        if n_xs == 0:
            raise RuntimeError(
                f"HEC-RAS 无法解析几何文件 (Schematic_XSCount=0)。"
                f"工作目录: {work_dir}"
            )

        # 3. 运行计算
        ras.Compute_HideComputationWindow()
        plan_names = ras.Plan_Names()
        if not plan_names or plan_names[0] == 0:
            raise RuntimeError("HEC-RAS 项目中没有找到计算方案")

        ras.Plan_SetCurrent(plan_names[1][0])
        ras.Compute_CurrentPlan()

        timeout = 300
        for i in range(timeout):
            time.sleep(1.0)
            if ras.Compute_Complete():
                break
        else:
            ras.Compute_Cancel()
            raise RuntimeError(f"HEC-RAS 计算超时 ({timeout}s)")

        ras.QuitRas()

        # 4. 检查计算结果 — 如果是纯稳态流, 使用此结果; 如果非恒定流已跑, 提取时间序列
        hdf_path = os.path.join(work_dir, "Yayao.p01.hdf")
        if not os.path.exists(hdf_path):
            raise RuntimeError("HEC-RAS 未生成输出 HDF5 文件")

        result = _extract_hecras_results(hdf_path, config, upstream_flow_m3s,
                                          downstream_stage_m, tidal_amplitude_m,
                                          tidal_period_h, duration_h, dt_s, work_dir)

        return result

    except Exception as e:
        raise RuntimeError(f"HEC-RAS 模拟失败: {e}. 工作目录: {work_dir}")


def _extract_hecras_results(hdf_path: str, config: YayaoConfig,
                             upstream_flow_m3s: float, downstream_stage_m: float,
                             tidal_amplitude_m: float, tidal_period_h: float,
                             duration_h: float, dt_s: float,
                             work_dir: str) -> HydroResult:
    """从 HEC-RAS 7.0 HDF5 输出文件提取水动力结果。"""
    import h5py

    with h5py.File(hdf_path, "r") as h5:
        results = h5["Results"]
        has_unsteady = "Unsteady" in results

        if has_unsteady:
            # 非恒定流
            base_path = "Results/Unsteady/Output/Output Blocks/Base Output"
            xs = h5[base_path + "/Cross Sections"]
            wl = np.array(xs["Water Surface"])
            q = np.array(xs["Flow"])
            vel = np.array(xs.get("Velocity", np.zeros_like(wl)))
            area = np.zeros_like(wl)

            if "Time" in h5[base_path]:
                t_hours = np.array(h5[base_path]["Time"], dtype=float) / 3600.0
            else:
                t_hours = np.linspace(0, duration_h, wl.shape[0])

            engine = "hecras_7.0_unsteady"
        else:
            # 稳态流
            base_path = "Results/Steady/Output/Output Blocks/Base Output/Steady Profiles"
            xs = h5[base_path + "/Cross Sections"]
            wl = np.array(xs["Water Surface"])
            q = np.array(xs["Flow"])
            # 流速在 Additional Variables/Velocity Total
            av = xs.get("Additional Variables")
            if av is not None and "Velocity Total" in av:
                vel = np.array(av["Velocity Total"])
            else:
                vel = np.zeros_like(wl)
            area = np.zeros_like(wl)

            t_hours = np.array([0.0])
            engine = "hecras_7.0_steady"

        # 桩号 — Geometry/Cross Sections/Attributes 是结构化 Dataset
        n_nodes = wl.shape[1]
        geo_attrs = h5["Geometry/Cross Sections/Attributes"]
        if hasattr(geo_attrs, "dtype") and geo_attrs.dtype.names is not None:
            # 结构化 dtype, 按字段名提取
            field_names = list(geo_attrs.dtype.names)
            # 优先匹配 RS (River Station), 然后是 Station, 最后是 River
            rs_field = None
            for fn in field_names:
                if fn == "RS" or fn == "River Station":
                    rs_field = fn
                    break
            if rs_field is None:
                for fn in field_names:
                    if "Station" in fn and "River" not in fn:
                        rs_field = fn
                        break
            if rs_field:
                raw = np.array(geo_attrs[rs_field]).flatten()
                # HEC-RAS 可能存储为字符串或数值
                if raw.dtype.kind in ("S", "U"):
                    stations = np.array([float(s.decode() if isinstance(s, bytes) else s) for s in raw])
                else:
                    stations = raw.astype(float)
            else:
                stations = np.linspace(config.length_m, 0, n_nodes)
        else:
            stations = np.linspace(config.length_m, 0, n_nodes)

        chainage = config.length_m - stations
        sort_idx = np.argsort(chainage)
        chainage = chainage[sort_idx]
        wl = wl[:, sort_idx]
        q = q[:, sort_idx]
        vel = vel[:, sort_idx]
        area = area[:, sort_idx]

    return HydroResult(
        t=t_hours,
        chainage=chainage,
        water_level=wl,
        discharge=q,
        velocity=vel,
        area=area,
        params={
            "upstream_flow_m3s": upstream_flow_m3s,
            "downstream_stage_m": downstream_stage_m,
            "tidal_amplitude_m": tidal_amplitude_m,
            "tidal_period_h": tidal_period_h,
            "duration_h": duration_h,
            "dt_s": dt_s,
            "engine": engine,
            "work_dir": work_dir,
        },
    )
