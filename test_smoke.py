"""Smoke test — 验证核心模块可正确导入和基本功能。"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")

_AIROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AIROOT)
os.chdir(_AIROOT)

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} — {detail}")

# 1. 模块导入
print("1. 模块导入")
try:
    import pandas as pd; import numpy as np
    import streamlit as st
    from water_quality_mcp.src.water_quality_mcp.analysis import (
        find_black_spots, calc_black_score_vectorized, data_summary,
        cross_validate_video_with_monitoring, build_comprehensive_summary,
    )
    from water_quality_mcp.src.water_quality_mcp.charts import (
        generate_time_series, generate_trace_map,
    )
    from water_quality_mcp.src.water_quality_mcp.state import WaterQualityState
    from river_model.config import YayaoConfig
    from river_model.cross_sections import generate_yayao_cross_sections
    from river_model.pure_py_hydro import solve_hydrodynamics
    from river_model.wq_engine import solve_water_quality
    from river_model.state import RiverModelState
    check("所有核心模块导入", True)
except Exception as e:
    check("所有核心模块导入", False, str(e))

# 2. 加载样本数据
print("2. 样本数据加载")
EXCEL = os.path.join(_AIROOT, "苗圃杯/无人船智能水质监测分析和因果溯源样本数据/河道巡航走航样本数据.xlsx")
if os.path.exists(EXCEL):
    try:
        df = pd.read_excel(EXCEL)
        col_map = {"氨氮(mg/L)": "ammonia_nitrogen", "溶解氧(mg/L)": "dissolved_oxygen",
                   "经度": "longitude", "纬度": "latitude", "化学需氧量(mg/L)": "cod",
                   "pH": "ph", "浑浊度(NTU)": "turbidity", "时间": "timestamp"}
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
        # Use synthetic datetime for test
        df['datetime'] = pd.date_range('2024-01-01', periods=len(df), freq='10s')
        check("加载 Excel", len(df) > 0, f"{len(df)} 行")
    except Exception as e:
        check("加载 Excel", False, str(e))
        df = None
else:
    check("加载 Excel", False, "样本文件不存在")
    df = None

# 3. 黑臭检测
print("3. 黑臭检测")
if df is not None:
    try:
        result_json, black = find_black_spots(df)
        result = json.loads(result_json)
        check("黑臭检测运行", "黑臭点数" in result or "error" not in result,
              f"黑臭点: {result.get('黑臭点数', 'N/A')}")
        check("聚类生成", result.get("聚类热点数", 0) > 0,
              f"聚类数: {result.get('聚类热点数', 0)}")
    except Exception as e:
        check("黑臭检测运行", False, str(e))

# 4. 河流模型
print("4. 河流模型")
try:
    config = YayaoConfig(n_cross_sections=20, mannings_n=0.028)
    cs_list = generate_yayao_cross_sections(20, 3200.0, config.bed_elevation_avg_m,
                                              config.longitudinal_slope, 40.0, mannings_n=0.028)
    hydro = solve_hydrodynamics(config, cs_list, upstream_flow_m3s=6.9,
        downstream_stage_m=0.83, tidal_amplitude_m=0.8, duration_h=3.0, dt_s=120.0)
    check("水动力模拟", not np.any(np.isnan(hydro.water_level)), "无 NaN")

    wq = solve_water_quality(config, hydro, pollutant_type="ammonia", load_kg=50.0,
        chainage_m=1600.0, duration_min=30.0, simulation_hours=6.0, dt_s=120.0)
    check("水质模拟", wq.ammonia is not None and not np.any(np.isnan(wq.ammonia)), "无 NaN")
except Exception as e:
    check("河流模型", False, str(e))

# 5. 图表生成
print("5. 图表生成")
if df is not None:
    try:
        ws = WaterQualityState()
        ws.df = df
        _, black = find_black_spots(df)
        ws.black_spots = black
        result = generate_time_series(ws, "test_time_series.png")
        check("时间序列图", os.path.exists("test_time_series.png"))
        result = generate_trace_map(ws, "test_trace_map.html")
        check("溯源地图", os.path.exists("test_trace_map.html"))
    except Exception as e:
        check("图表生成", False, str(e))
    finally:
        for f in ["test_time_series.png", "test_trace_map.html"]:
            if os.path.exists(f):
                os.remove(f)

print(f"\n{'='*40}")
print(f"结果: {passed} 通过, {failed} 失败, {passed+failed} 总计")
if failed > 0:
    print("WARN: Some tests failed, see above")
    sys.exit(1)
else:
    print("OK: All core functions working")
