import streamlit as st
import pandas as pd
import numpy as np
import os, json, base64, time, tempfile, asyncio, math
from datetime import timedelta
from openai import OpenAI
from sklearn.cluster import DBSCAN
import cv2
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import matplotlib.font_manager as fm
import requests
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from river_model.tools import (
    tool_river_init, tool_river_hydro_simulation, tool_river_pollution_event,
    tool_river_concentration_profile, tool_river_compare_scenarios, tool_river_export_data,
)
from river_model.state import RiverModelState

st.set_page_config(page_title="水质监测溯源助手", layout="wide")

# 星图地球数据云 Token — 优先从环境变量读取, 否则使用用户配置
def _get_starcloud_token():
    token = os.environ.get("CLOUD_TOKEN", "")
    if token and "your-" not in token:
        return token
    return st.session_state.get("_starcloud_token", "")

# 初始化 Session State
if "df" not in st.session_state: st.session_state.df = None
if "black_spots" not in st.session_state: st.session_state.black_spots = None
if "video_frames" not in st.session_state: st.session_state.video_frames = []
if "messages" not in st.session_state: st.session_state.messages = []
if "full_messages" not in st.session_state: st.session_state.full_messages = []
if "analysis_done" not in st.session_state: st.session_state.analysis_done = False
if "charts_generated" not in st.session_state: st.session_state.charts_generated = False
if "anim_html" not in st.session_state: st.session_state.anim_html = None
if "anim_file_size" not in st.session_state: st.session_state.anim_file_size = 0
if "cross_validation_result" not in st.session_state: st.session_state.cross_validation_result = None
if "chart_captions" not in st.session_state: st.session_state.chart_captions = {}
if "river_state" not in st.session_state: st.session_state.river_state = RiverModelState()
if "font_prop" not in st.session_state:
    font_prop = None
    for fp in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]:
        if os.path.exists(fp): font_prop = fm.FontProperties(fname=fp); break
    st.session_state.font_prop = font_prop
    plt.rcParams['font.family'] = font_prop.get_name() if font_prop else 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

# ---------- API 密钥管理 ----------

def _load_api_keys_from_files():
    """从 .env 和 .mcp.json 加载已保存的 API key 到 session_state。"""
    keys = {"deepseek": "", "amap": "", "starcloud": ""}
    # 读 .env
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DASHSCOPE_API_KEY="):
                    v = line.split("=", 1)[1].strip()
                    if v and "your-" not in v:
                        keys["deepseek"] = v
    # 读 .mcp.json
    mcp_path = os.path.join(os.path.dirname(__file__), ".mcp.json")
    if os.path.exists(mcp_path):
        try:
            with open(mcp_path, encoding="utf-8") as f:
                cfg = json.load(f)
            env_vars = cfg.get("mcpServers", {}).get("water-quality-mcp", {}).get("env", {})
            for k in ["AMAP_KEY", "DEEPSEEK_API_KEY", "CLOUD_TOKEN"]:
                v = (env_vars.get(k, "") or "").strip()
                if v and "your-" not in v:
                    if k == "AMAP_KEY":
                        keys["amap"] = v
                    elif k == "DEEPSEEK_API_KEY":
                        keys["deepseek"] = v or keys["deepseek"]
                    elif k == "CLOUD_TOKEN":
                        keys["starcloud"] = v
        except Exception:
            pass
    return keys

def _save_api_keys_to_files(deepseek_key, amap_key, starcloud_token):
    """保存 API key 到 .env 和 .mcp.json 文件。"""
    import os as _os
    root = os.path.dirname(__file__)
    # 写 .env
    with open(_os.path.join(root, ".env"), "w", encoding="utf-8") as f:
        f.write(f"DASHSCOPE_API_KEY={deepseek_key}\n")
    # 写 .mcp.json
    mcp_config = {
        "mcpServers": {
            "water-quality-mcp": {
                "command": "python",
                "args": ["-m", "water_quality_mcp.server"],
                "env": {
                    "AMAP_KEY": amap_key or "",
                    "CLOUD_TOKEN": starcloud_token or "",
                    "DEEPSEEK_API_KEY": deepseek_key or "",
                }
            }
        }
    }
    with open(_os.path.join(root, ".mcp.json"), "w", encoding="utf-8") as f:
        json.dump(mcp_config, f, indent=2, ensure_ascii=False)

def _show_setup_wizard():
    """首次运行或 key 缺失时弹出 API 密钥配置界面。"""
    keys = _load_api_keys_from_files()
    has_keys = bool(keys["deepseek"])

    if has_keys and "_setup_done" not in st.session_state:
        st.session_state._setup_done = True
        st.session_state._deepseek_key = keys["deepseek"]
        st.session_state._amap_key = keys["amap"]
        st.session_state._starcloud_token = keys["starcloud"]
        return True

    if st.session_state.get("_setup_done"):
        return True

    # 显示配置向导
    st.markdown("""
    ## 🔐 首次运行 — 配置 API 密钥

    本工具依赖以下第三方服务，请填写您的 API 密钥后点击"保存并开始使用"。

    | 服务 | 用途 | 获取地址 |
    |------|------|---------|
    | **DeepSeek**（必填） | LLM 自动分析 & 溯源报告 | [platform.deepseek.com](https://platform.deepseek.com) |
    | 高德地图（可选） | 逆地理编码 & POI 搜索 | [console.amap.com](https://console.amap.com) |
    | 星图地球（可选） | 卫星影像底图 | [datacloud.geovisearth.com](https://datacloud.geovisearth.com) |
    """)

    with st.form("setup_form"):
        dk = st.text_input("DeepSeek API Key *", type="password",
                          value=st.session_state.get("_deepseek_key", ""),
                          placeholder="sk-...")
        ak = st.text_input("高德地图 API Key（可选）", type="password",
                          value=st.session_state.get("_amap_key", ""),
                          placeholder="留空则跳过地理相关功能")
        sc = st.text_input("星图地球 Token（可选）", type="password",
                          value=st.session_state.get("_starcloud_token", ""),
                          placeholder="留空则使用默认瓦片")
        submitted = st.form_submit_button("💾 保存并开始使用", type="primary", use_container_width=True)

        if submitted:
            if not dk.strip():
                st.error("DeepSeek API Key 为必填项")
                return False
            _save_api_keys_to_files(dk.strip(), ak.strip(), sc.strip())
            st.session_state._setup_done = True
            st.session_state._deepseek_key = dk.strip()
            st.session_state._amap_key = ak.strip()
            st.session_state._starcloud_token = sc.strip()
            st.rerun()
    return False

# ---------- MCP 客户端 ----------
class MCPClient:
    def __init__(self): self.process = None
    async def start(self):
        self.process = await asyncio.create_subprocess_exec("water-quality-mcp", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    async def stop(self):
        if self.process: self.process.terminate(); await self.process.wait()
    async def call_tool(self, tool_name, arguments=None):
        if not self.process: await self.start()
        req = {"jsonrpc":"2.0","method":"tools/call","params":{"name":tool_name,"arguments":arguments or {}},"id":1}
        self.process.stdin.write((json.dumps(req)+"\n").encode()); await self.process.stdin.drain()
        line = await self.process.stdout.readline()
        if line:
            res = json.loads(line.decode())
            return res.get("result", res.get("error","无响应"))
        return "无响应"

# ---------- 数据加载（已修复 pH 0 值问题 + 自动清除旧图表）----------
def load_data(uploaded_file):
    if uploaded_file is None: return "请先上传Excel或CSV文件"
    try:
        fn = uploaded_file.name.lower()
        if fn.endswith('.csv'):
            try: df = pd.read_csv(uploaded_file, encoding='utf-8')
            except: df = pd.read_csv(uploaded_file, encoding='gbk')
        elif fn.endswith(('.xls','.xlsx')): df = pd.read_excel(uploaded_file)
        else: return "不支持的文件格式"
    except Exception as e: return f"读取失败: {e}"

    # ---- 重置所有分析状态 ----
    st.session_state.black_spots = None
    st.session_state.video_frames = []
    st.session_state.cross_validation_result = None
    st.session_state.chart_captions = {}
    st.session_state.anim_html = None
    st.session_state.anim_file_size = 0
    st.session_state.analysis_done = False
    st.session_state.charts_generated = False
    st.session_state.messages = []
    st.session_state.full_messages = []
    st.session_state.river_state = RiverModelState()

    # ---- 清除所有旧图表文件 ----
    _CHART_FILES = [
        "time_series.png", "trace_map.html",
        "cross_validation.png", "correlation_heatmap.png",
        "river_hydro_profile.png", "river_discharge_profile.png",
        "river_wq_contour.png", "river_wq_profile.png",
        "river_wq_timeseries.png", "river_dashboard.png",
        "river_pollution_animation.html", "_uploaded_data.xlsx",
    ]
    for f in _CHART_FILES:
        if os.path.exists(f):
            try: os.remove(f)
            except OSError: pass

    # ---- 保存上传数据副本供 build_trace_animation 使用 ----
    try:
        df.to_excel("_uploaded_data.xlsx", index=False)
    except Exception:
        pass
    col_map = {
        '时间':'timestamp','氨氮(mg/L)':'ammonia_nitrogen','溶解氧(mg/L)':'dissolved_oxygen',
        '水温(°C)':'water_temperature','经度':'longitude','纬度':'latitude','电导率':'conductivity',
        '化学需氧量(mg/L)':'cod','总有机碳(mg/L)':'toc','pH':'ph','浑浊度(NTU)':'turbidity'
    }
    df.rename(columns=col_map, inplace=True)
    ts = df['timestamp'].astype(str).str.strip().str.replace('\u3000',' ').str.replace('：',':')
    dt = pd.to_datetime(ts, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    mask = dt.isna()
    if mask.any():
        short = ts[mask].str.slice(0,16)
        dt[mask] = pd.to_datetime(short, format='%Y-%m-%d %H:%M', errors='coerce')
    df['datetime'] = dt
    df.dropna(subset=['datetime'], inplace=True)
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 数值清洗
    numeric_cols = ['ammonia_nitrogen','dissolved_oxygen','water_temperature','cod','turbidity','ph']
    for col in numeric_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

    # 剔除物理上不可能的0值（pH、溶解氧、水温）
    for col in ['ph','dissolved_oxygen','water_temperature']:
        if col in df.columns: df[col] = df[col].replace(0, np.nan)

    # 范围限定
    limits = {'ammonia_nitrogen':(0,1000),'dissolved_oxygen':(0,20),'water_temperature':(0,45),
              'cod':(0,500),'turbidity':(0,2000),'ph':(3,12)}
    for col, (lo,hi) in limits.items():
        if col in df.columns: df[col] = df[col].where((df[col]>=lo)&(df[col]<=hi), np.nan)

    # 线性插值
    for col in numeric_cols:
        if col in df.columns and df[col].isna().sum()>0:
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
    for col in ['longitude','latitude']:
        if col in df.columns: df[col] = df[col].ffill().bfill()

    st.session_state.df = df
    # 返回验证信息
    report = [f"✅ 数据加载完成，共 {len(df)} 条记录"]
    for col in ['ammonia_nitrogen','dissolved_oxygen','turbidity','ph']:
        valid = df[col].dropna()
        report.append(f"   {col}: 有效值={len(valid)}, 均值={valid.mean():.2f}, 最小={valid.min():.2f}, 最大={valid.max():.2f}")
    return "\n".join(report)

def extract_video(video_file, interval=10):
    if video_file is None: st.session_state.video_frames = []; return "未上传视频文件"
    # 分块写入临时文件，避免大视频 OOM
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        vpath = tmp.name
        chunk_size = 64 * 1024 * 1024  # 64MB 分块
        while True:
            chunk = video_file.read(chunk_size)
            if not chunk:
                break
            tmp.write(chunk)
    file_size_mb = os.path.getsize(vpath) / (1024 * 1024)
    if file_size_mb < 0.01:
        os.unlink(vpath)
        return "视频文件为空，请检查上传是否完整"
    # 抑制 FFmpeg H.264 解码警告输出
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
    cap = cv2.VideoCapture(vpath, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        os.unlink(vpath)
        return f"无法解码视频（{file_size_mb:.0f}MB），请检查编码格式是否为 H.264/H.265"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frame = max(1, int(fps * interval))
    count, frames, skipped = 0, [], 0
    while True:
        try:
            ret, frame = cap.read()
        except Exception:
            skipped += 1; count += 1
            continue
        if not ret:
            break
        if count % interval_frame == 0:
            # 校验帧是否有效（非全灰/全黑损坏帧）
            if frame is not None and frame.size > 0 and frame.mean() > 1.0:
                _, buf = cv2.imencode('.jpg', frame)
                frames.append({'time_sec': round(count / fps, 1),
                               'image_base64': base64.b64encode(buf).decode()})
            else:
                skipped += 1
        count += 1
    cap.release(); os.unlink(vpath)
    st.session_state.video_frames = frames
    return f"视频抽帧完成，共 {len(frames)} 帧（总 {total_frames} 帧，跳过 {skipped} 个损坏帧）"

# ---------- 工具函数 ----------
def tool_data_summary():
    df = st.session_state.df
    if df is None: return "数据未加载"
    s = df.agg({'ammonia_nitrogen':['mean','max'],'dissolved_oxygen':['mean','min'],'turbidity':['mean','max'],'ph':['mean','min','max'],'cod':['mean','max']})
    return json.dumps({"记录数":len(df),"时间范围":f"{df['datetime'].min()} ~ {df['datetime'].max()}",
                       "氨氮(mg/L)":{"均值":round(s['ammonia_nitrogen']['mean'],2),"最大":round(s['ammonia_nitrogen']['max'],2)},
                       "溶解氧(mg/L)":{"均值":round(s['dissolved_oxygen']['mean'],2),"最小":round(s['dissolved_oxygen']['min'],2)},
                       "浑浊度(NTU)":{"均值":round(s['turbidity']['mean'],1),"最大":round(s['turbidity']['max'],1)},
                       "pH":{"均值":round(s['ph']['mean'],2),"最小":round(s['ph']['min'],2),"最大":round(s['ph']['max'],2)}}, ensure_ascii=False)

# ---------- 黑点分析辅助函数 ----------
def tool_find_black_spots():
    """黑臭检测 - 委托给 MCP analysis 模块统一实现。"""
    from water_quality_mcp.src.water_quality_mcp.analysis import find_black_spots as _find
    result, black = _find(st.session_state.df)
    if black is not None:
        st.session_state.black_spots = black
    return result

def tool_video_analysis():
    frames = st.session_state.video_frames
    if not frames: return "无视频帧"
    results = []
    for f in frames[:20]:
        buf = base64.b64decode(f['image_base64']); img = cv2.imdecode(np.frombuffer(buf,np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s,v = hsv[:,:,1].mean(), hsv[:,:,2].mean(); lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        turb = "清澈" if lap>200 else ("轻微浑浊" if lap>100 else ("中度浑浊" if lap>30 else "重度浑浊"))
        col = "疑似黑臭" if (s<40 and v<70) else "正常"
        risk = "需关注" if turb in ["中度浑浊","重度浑浊"] or col=="疑似黑臭" else "正常"
        results.append(f"时间{f['time_sec']}s: 浑浊度={turb}, 颜色={col}, 风险={risk}")
    return "\n".join(results)

def tool_cross_validate_video(time_offset_sec=0, time_window_sec=30):
    from water_quality_mcp.src.water_quality_mcp.analysis import cross_validate_video_with_monitoring
    from water_quality_mcp.src.water_quality_mcp.state import WaterQualityState
    ws = WaterQualityState()
    ws.df = st.session_state.df
    ws.video_frames = st.session_state.video_frames
    ws.black_spots = st.session_state.black_spots
    result = cross_validate_video_with_monitoring(ws, time_offset_sec, time_window_sec)
    st.session_state.cross_validation_result = json.loads(result) if isinstance(result, str) else result
    return result

def tool_generate_cross_validation_chart(time_offset_sec=0):
    from water_quality_mcp.src.water_quality_mcp.charts import generate_cross_validation_chart
    from water_quality_mcp.src.water_quality_mcp.state import WaterQualityState
    ws = WaterQualityState()
    ws.df = st.session_state.df
    ws.video_frames = st.session_state.video_frames
    ws.black_spots = st.session_state.black_spots
    result = generate_cross_validation_chart(ws, "cross_validation.png", time_offset_sec)
    _build_cross_validation_caption()
    return result

def tool_get_comprehensive_summary():
    from water_quality_mcp.src.water_quality_mcp.analysis import build_comprehensive_summary
    from water_quality_mcp.src.water_quality_mcp.state import WaterQualityState
    ws = WaterQualityState()
    ws.df = st.session_state.df
    ws.black_spots = st.session_state.black_spots
    ws.video_frames = st.session_state.video_frames
    ws.cross_validation_result = st.session_state.cross_validation_result
    return build_comprehensive_summary(ws, st.session_state.river_state)

def search_nearby_pollution_sources(lat, lon, amap_key, radius=500):
    keywords = ["工厂","养殖场","排污口","化工厂"]
    all_pois = []
    for kw in keywords:
        url = "https://restapi.amap.com/v3/place/around"
        params = {"location":f"{lon},{lat}","keywords":kw,"radius":radius,"key":amap_key,"offset":15}
        try:
            resp = requests.get(url, params=params, timeout=8); data = resp.json()
            if data.get("status")=="1":
                for p in data.get("pois",[]):
                    loc = p.get("location","").split(",")
                    all_pois.append({"name":p.get("name",""),"type":p.get("type",""),"address":p.get("address",""),
                                     "distance":p.get("distance",""),"lat":float(loc[1]) if len(loc)==2 else lat,"lon":float(loc[0]) if len(loc)==2 else lon})
        except: continue
    # 按距离排序
    def dist_num(poi):
        d = poi.get('distance')
        if d is None:
            return 999999
        digits = ''.join(filter(str.isdigit, str(d)))
        return int(digits) if digits else 999999
    all_pois.sort(key=dist_num)
    return all_pois

def tool_generate_trace_map(amap_key=None, cloud_token=None):
    from water_quality_mcp.src.water_quality_mcp.analysis import get_severity_info, get_cluster_extent
    df = st.session_state.df; black = st.session_state.black_spots
    if df is None or black is None: return "数据或黑点未就绪"
    coords = df[['latitude','longitude']].dropna().values
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=15, tiles=None)
    # —— 底图（多源 LayerControl 切换，应对部分服务器在国内不可达）——
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png',
        name='CartoDB Voyager (WGS-84 推荐)',
        attr='CartoDB', max_zoom=19, show=True,
        subdomains='abc',
    ).add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        name='Esri 卫星图 (WGS-84 无偏移)',
        attr='Esri', max_zoom=19, show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='OSM 街道图 (WGS-84)',
        attr='OSM', max_zoom=19, show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
        name='CartoDB Light (WGS-84)',
        attr='CartoDB', max_zoom=19, show=False,
        subdomains='abc',
    ).add_to(m)
    token = cloud_token or _get_starcloud_token()
    if token and "你的星图" not in token:
        geo_tile_url = f'https://tiles1.geovisearth.com/base/v1/img/{{z}}/{{x}}/{{y}}?format=webp&tmsIds=w&token={token}'
        folium.TileLayer(
            tiles=geo_tile_url,
            name='星图地球卫星图 (国内可能GCJ-02偏移)',
            attr='星图地球', max_zoom=18, show=False,
        ).add_to(m)
    if amap_key:
        tile_url = 'https://webst{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}'
        folium.TileLayer(
            tiles=tile_url,
            name='高德卫星图 (GCJ-02 有偏移)',
            attr='高德', max_zoom=18, show=False,
            subdomains=['01', '02', '03', '04'],
        ).add_to(m)
    # 轨迹 + 起点终点
    folium.PolyLine(coords, weight=2, color='blue', opacity=0.5).add_to(m)
    folium.map.Marker(coords[0], icon=folium.DivIcon(html='<div style="color:green;font-weight:bold;">起点</div>')).add_to(m)
    folium.map.Marker(coords[-1], icon=folium.DivIcon(html='<div style="color:red;font-weight:bold;">终点</div>')).add_to(m)
    # 热力图
    HeatMap(black[['latitude','longitude']].values, radius=20, blur=10).add_to(m)
    # 采样点（每30个）
    sample = df.iloc[::30]
    for _, row in sample.iterrows():
        nh3 = f"{row['ammonia_nitrogen']:.1f}" if pd.notna(row.get('ammonia_nitrogen')) else '--'
        do = f"{row['dissolved_oxygen']:.1f}" if pd.notna(row.get('dissolved_oxygen')) else '--'
        cod = f"{row['cod']:.0f}" if pd.notna(row.get('cod')) else '--'
        turb = f"{row['turbidity']:.0f}" if pd.notna(row.get('turbidity')) else '--'
        popup_html = f"<b>{row['datetime']}</b><br>氨氮:{nh3} mg/L<br>DO:{do} mg/L<br>COD:{cod} mg/L<br>浊度:{turb} NTU"
        folium.CircleMarker([row['latitude'], row['longitude']], radius=3, color='darkblue', fill=True, fill_opacity=0.6,
                            popup=folium.Popup(popup_html, max_width=220)).add_to(m)
    # 聚类渲染（按严重等级着色 + 虚线范围圆 + 每聚类搜 POI）
    for cid in sorted(black['cluster_id'].unique()):
        if cid == -1: continue
        cluster = black[black['cluster_id'] == cid]
        avg_score = float(cluster['black_score'].mean())
        _, sev_label, sev_color = get_severity_info(avg_score)
        # 聚类范围虚线圆
        clat, clon, extent_r = get_cluster_extent(cluster)
        folium.Circle([clat, clon], radius=extent_r, color=sev_color, weight=1.5, fill=True,
                      fill_opacity=0.06, dash_array='5,5').add_to(m)
        # 聚类中心实心圆
        rad = max(12, min(30, len(cluster) * 1.5))
        t_start = cluster['datetime'].min().strftime('%H:%M')
        t_end = cluster['datetime'].max().strftime('%H:%M')
        nh3_m = f"{cluster['ammonia_nitrogen'].mean():.2f}" if 'ammonia_nitrogen' in cluster.columns else '--'
        do_m = f"{cluster['dissolved_oxygen'].mean():.2f}" if 'dissolved_oxygen' in cluster.columns else '--'
        popup_html = (f"<b>热点 {cid} — {sev_label}</b><br>"
                      f"采样点: {len(cluster)} 个<br>时间: {t_start} ~ {t_end}<br>"
                      f"评分均值: {avg_score:.2f}<br>氨氮均值: {nh3_m} mg/L<br>DO均值: {do_m} mg/L")
        if 'cod' in cluster.columns:
            popup_html += f"<br>COD均值: {cluster['cod'].mean():.0f} mg/L"
        if 'turbidity' in cluster.columns:
            popup_html += f"<br>浊度均值: {cluster['turbidity'].mean():.0f} NTU"
        if 'ph' in cluster.columns:
            popup_html += f"<br>pH均值: {cluster['ph'].mean():.1f}"
        folium.CircleMarker([clat, clon], radius=rad, color=sev_color, fill=True, fill_opacity=0.55,
                            weight=3, popup=folium.Popup(popup_html, max_width=280)).add_to(m)
        # 对每个聚类搜索周边污染源
        if amap_key:
            pois = search_nearby_pollution_sources(clat, clon, amap_key)
            for poi in pois[:5]:
                folium.Marker([poi['lat'], poi['lon']],
                              popup=f"{poi['name']}({poi['type']})<br>{poi.get('address','')}",
                              icon=folium.Icon(color='black', icon='industry')).add_to(m)
    # 污染最严重 Top-5 个体点位标记
    top5 = black.nlargest(5, 'black_score')
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        sev_color = row['severity_color']
        _, sev_label, _ = get_severity_info(row['black_score'])
        icon_html = (
            f'<div style="background:{sev_color};color:#fff;width:26px;height:26px;'
            f'border-radius:50%;text-align:center;line-height:26px;'
            f'font-weight:bold;font-size:14px;border:2px solid #fff;'
            f'box-shadow:0 0 8px rgba(0,0,0,0.6);">{i}</div>'
        )
        t_str = row['datetime'].strftime('%Y-%m-%d %H:%M:%S')
        nh3_v = f"{row['ammonia_nitrogen']:.2f}" if pd.notna(row.get('ammonia_nitrogen')) else '--'
        do_v = f"{row['dissolved_oxygen']:.2f}" if pd.notna(row.get('dissolved_oxygen')) else '--'
        cod_v = f"{row['cod']:.0f}" if pd.notna(row.get('cod')) else '--'
        turb_v = f"{row['turbidity']:.0f}" if pd.notna(row.get('turbidity')) else '--'
        ph_v = f"{row['ph']:.1f}" if pd.notna(row.get('ph')) else '--'
        popup_html = (
            f"<b>第{i}严重点位 — {sev_label}</b><br>"
            f"评分: {row['black_score']:.2f}<br>"
            f"时间: {t_str}<br>"
            f"氨氮: {nh3_v} mg/L | DO: {do_v} mg/L<br>"
            f"COD: {cod_v} mg/L | 浊度: {turb_v} NTU<br>"
            f"pH: {ph_v}"
        )
        folium.Marker(
            [row['latitude'], row['longitude']],
            icon=folium.DivIcon(html=icon_html),
            popup=folium.Popup(popup_html, max_width=280)
        ).add_to(m)

    # 严重程度图例 (GB3838 V类 + 内梅罗指数)
    legend_html = '''
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
        background:rgba(255,255,255,0.92);padding:10px 14px;
        border-radius:6px;font-size:12px;line-height:1.7;
        box-shadow:0 0 6px rgba(0,0,0,0.25);">
    <b>GB3838-2002 V类 污染等级 (内梅罗指数)</b><br>
    <span style="color:#2E8B57;">&#9679;</span> 达V类<br>
    <span style="color:#FFA500;">&#9679;</span> 轻污染 (超V类)<br>
    <span style="color:#FF4500;">&#9679;</span> 污染 (超V类2-3倍)<br>
    <span style="color:#DC143C;">&#9679;</span> 重污染 (超V类3-5倍)<br>
    <span style="color:#8B0000;">&#9679;</span> 严重污染 (超V类&gt;5倍)<br>
    <span style="color:gray;">---</span> 虚线圆 = 聚类范围<br>
    <span style="background:#DC143C;color:#fff;border-radius:50%;display:inline-block;
    width:14px;height:14px;text-align:center;line-height:14px;font-size:10px;">1</span>
    数字标记 = 污染最严重点位 (Top 5)
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LatLngPopup().add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save('trace_map.html'); st.session_state.charts_generated = True
    # 生成图表说明
    _build_trace_map_caption(black, amap_key)
    # 返回数据摘要供 LLM 消费
    black = st.session_state.black_spots
    summary = {"图表": "交互式溯源地图已生成", "文件": "trace_map.html"}
    if black is not None and len(black) > 0:
        clusters_out = []
        for cid in sorted(black['cluster_id'].unique()):
            if cid == -1: continue
            c = black[black['cluster_id'] == cid]
            clusters_out.append({
                "聚类ID": int(cid), "点数": len(c), "内梅罗均值": round(float(c['black_score'].mean()), 2),
                "主导黑臭等级": c['severity_label'].mode().iloc[0] if len(c) > 0 else "",
                "中心经纬度": f"{c['latitude'].mean():.5f},{c['longitude'].mean():.5f}",
                "氨氮均值": round(float(c['ammonia_nitrogen'].mean()), 1),
                "DO均值": round(float(c['dissolved_oxygen'].mean()), 2),
            })
        summary["聚类概览"] = clusters_out
        top3 = black.nlargest(3, 'black_score')
        summary["最严重3点"] = [{
            "排名": i+1, "经纬度": f"{r['latitude']:.5f},{r['longitude']:.5f}",
            "内梅罗指数": round(float(r['black_score']), 2), "黑臭等级": r['severity_label'],
            "氨氮": round(float(r['ammonia_nitrogen']), 1), "DO": round(float(r['dissolved_oxygen']), 2),
        } for i, (_, r) in enumerate(top3.iterrows())]
    return json.dumps(summary, ensure_ascii=False)

# ---------- 图表说明自动生成 ----------

def _build_time_series_caption(df, black):
    """根据数据自动生成时间序列图说明"""
    if df is None:
        st.session_state.chart_captions["time_series"] = "时间序列图：展示走航监测过程中氨氮、溶解氧、浊度、pH 四项指标随时间的变化趋势。"
        return
    t_start = df['datetime'].min().strftime('%Y-%m-%d %H:%M')
    t_end = df['datetime'].max().strftime('%H:%M')
    nh3_peak = df['ammonia_nitrogen'].max() if 'ammonia_nitrogen' in df.columns else 0
    do_min = df['dissolved_oxygen'].min() if 'dissolved_oxygen' in df.columns else 0
    parts = [
        f"**时间序列图** — 监测时段 {t_start} ~ {t_end}，共 {len(df)} 条记录。",
        f"氨氮峰值 {nh3_peak:.1f} mg/L，溶解氧最低 {do_min:.2f} mg/L。",
    ]
    if black is not None and len(black) > 0:
        n_black = len(black)
        sev_dist = black['severity_label'].value_counts().to_dict()
        mild = sev_dist.get('轻度黑臭', 0)
        severe = sev_dist.get('重度黑臭', 0)
        parts.append(f"检出黑臭点位 {n_black} 个（轻度 {mild}，重度 {severe}），散点颜色标注于图中。")
    parts.append("红色虚线 = GB3838-2002 V类标准限值（氨氮≤2.0 mg/L，DO≥2.0 mg/L）。")
    st.session_state.chart_captions["time_series"] = " ".join(parts)

def _build_trace_map_caption(black, amap_key):
    """根据数据自动生成溯源地图说明"""
    if black is None or len(black) == 0:
        st.session_state.chart_captions["trace_map"] = "交互式溯源地图：显示走航轨迹、采样点分布和污染热力图。"
        return
    clusters = [c for c in sorted(black['cluster_id'].unique()) if c != -1]
    n_clusters = len(clusters)
    worst_sev = black['severity_label'].mode().iloc[0] if len(black) > 0 else "未知"
    parts = [
        f"**交互式溯源地图** — DBSCAN 聚类识别 {n_clusters} 个污染热点区域，最严重等级为「{worst_sev}」。",
        f"虚线圆 = 聚类影响范围，实心圆 = 聚类中心（大小反映样本密度），数字标记 = Top 5 最严重点位。",
    ]
    if amap_key:
        parts.append("黑色工厂图标 = 周边疑似污染源（高德POI搜索）。")
    parts.append("底图支持切换高德卫星图（GCJ-02坐标系，有偏移），点击标记可查看详情。")
    st.session_state.chart_captions["trace_map"] = " ".join(parts)

def _build_cross_validation_caption():
    """根据交叉印证结果生成说明"""
    cv = st.session_state.cross_validation_result
    if cv is None:
        st.session_state.chart_captions["cross_validation"] = "视频-传感器交叉印证图：上图为视频 Laplacian 方差与传感器浊度对比，下图为 HSV 亮度与氨氮对比。"
        return
    parts = ["**交叉印证图** — 上：视频 Laplacian 方差（绿线，越高越清晰）vs 传感器浊度（蓝线，越高越浑浊），预期负相关。"]
    parts.append("下：视频 HSV 亮度（绿线，越低越暗）vs 传感器氨氮（红线），预期负相关（黑臭水体色深氨氮高）。")
    corr = cv.get("correlation_pairs", cv.get("correlations", {}))
    if corr:
        if isinstance(corr, dict):
            for k, v in corr.items():
                parts.append(f"{k}: r={v:.3f}" if isinstance(v, (int, float)) else f"{k}: {v}")
        elif isinstance(corr, list):
            for item in corr[:4]:
                if isinstance(item, dict):
                    parts.append(f"{item.get('pair','?')}: r={item.get('pearson_r','?')}")
    agree = cv.get("agreement", cv.get("black_odor_agreement", {}))
    if agree and isinstance(agree, dict):
        rate = agree.get("agreement_rate", agree.get("一致率", ""))
        if rate:
            parts.append(f"黑臭判定一致率: {rate}" if isinstance(rate, str) else f"黑臭判定一致率: {rate:.1%}")
    st.session_state.chart_captions["cross_validation"] = " ".join(parts)

def _build_correlation_heatmap_caption():
    """根据交叉印证结果生成相关系数热力图说明"""
    cv = st.session_state.cross_validation_result
    if cv is None:
        st.session_state.chart_captions["correlation_heatmap"] = "相关系数热力图：展示视频视觉指标（Laplacian方差/HSV饱和度/亮度）与传感器指标（浊度/氨氮/DO/COD）的 Pearson 相关系数矩阵。"
        return
    corr = cv.get("correlation_pairs", cv.get("correlations", {}))
    parts = ["**相关系数热力图** — 视频视觉指标 × 传感器指标的 Pearson r 矩阵。"]
    if isinstance(corr, dict):
        best = max(corr.items(), key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0) if corr else None
        if best:
            parts.append(f"最强相关: {best[0]} (r={best[1]:.3f})。")
    elif isinstance(corr, list) and corr:
        best = max(corr, key=lambda x: abs(x.get('pearson_r', 0)) if isinstance(x, dict) else 0)
        if isinstance(best, dict):
            parts.append(f"最强相关: {best.get('pair','?')} (r={best.get('pearson_r','?')})。")
    parts.append("红色=正相关，蓝色=负相关，颜色越深相关性越强。")
    st.session_state.chart_captions["correlation_heatmap"] = " ".join(parts)

def _get_wq_conc(wq):
    """获取 WQResult 中的主要污染物浓度数组。"""
    if wq.ammonia is not None:
        return wq.ammonia
    if wq.cbod is not None:
        return wq.cbod
    return wq.dissolved_oxygen

def _build_river_animation_caption(anim_path=None):
    """根据河道模拟结果生成污染溯源动画说明"""
    rs = st.session_state.river_state
    parts = ["**污染溯源动画** — 基于真实走航数据 + Preissmann 非恒定流模型 + ADR 水质模型生成。"]
    if rs.has_hydro():
        h = rs.hydro_result
        parts.append(f"模拟时长 {h.t[-1]:.0f}h，{len(h.chainage)} 个断面，潮汐振幅 {h.params.get('tidal_amplitude_m',0.8):.2f}m。")
    if rs.has_wq():
        wq = rs.wq_result
        conc = _get_wq_conc(wq)
        peak_c = float(np.max(conc))
        peak_t = float(wq.t[int(np.argmax(np.max(conc, axis=1)))])
        parts.append(f"污染物峰值浓度 {peak_c:.2f} mg/L，出现在模拟第 {peak_t:.1f}h。")
    parts.append("色标基于 GB3838-2002 五类水质标准，支持播放/暂停、时间滑块和速度调节。")
    st.session_state.chart_captions["river_animation"] = " ".join(parts)

def _render_river_chart_caption(fname, title, rs):
    """为河道模型图表渲染数据驱动的说明"""
    hydro = rs.hydro_result if rs.has_hydro() else None
    wq = rs.wq_result if rs.has_wq() else None
    if "hydro_profile" in fname:
        if hydro is not None:
            z_max = float(np.max(hydro.water_level)); z_min = float(np.min(hydro.water_level))
            st.caption(f"沿程水位剖面 — 最高水位 {z_max:.3f}m，最低 {z_min:.3f}m，{len(hydro.chainage)} 个断面。表征潮汐影响下的水面线变化。")
        else:
            st.caption(f"沿程水位剖面 — 基于 Saint-Venant 方程 Preissmann 隐式格式求解，展示各断面水位沿程分布。")
    elif "discharge_profile" in fname:
        if hydro is not None:
            q_max = float(np.max(np.abs(hydro.discharge)))
            st.caption(f"沿程流量剖面 — 最大流量 {q_max:.2f} m³/s，正值为顺流向（落潮），负值为逆流向（涨潮）。")
        else:
            st.caption(f"沿程流量剖面 — 展示各断面流量沿程分布，受潮汐边界条件影响。")
    elif "wq_contour" in fname:
        if wq is not None:
            conc = _get_wq_conc(wq)
            peak_c = float(np.max(conc)); peak_t = float(wq.t[int(np.argmax(np.max(conc, axis=1)))])
            st.caption(f"浓度时空分布 — 峰值 {peak_c:.2f} mg/L，出现在 {peak_t:.1f}h。横轴为时间，纵轴为桩号，颜色深浅反映污染物浓度。")
        else:
            st.caption(f"浓度时空分布 — 横轴时间、纵轴桩号，颜色深浅反映污染物浓度沿河道随时间的变化。")
    elif "wq_profile" in fname:
        if wq is not None:
            st.caption(f"浓度沿程剖面 — 展示特定时刻各断面污染物浓度，用于识别污染峰值位置和影响范围。")
        else:
            st.caption(f"浓度沿程剖面 — 展示特定时刻各断面污染物浓度沿程分布。")
    elif "wq_timeseries" in fname:
        if wq is not None:
            st.caption(f"浓度过程线 — 展示特定断面污染物浓度随时间变化，反映污染团经过时的浓度升降过程。")
        else:
            st.caption(f"浓度过程线 — 展示特定断面污染物浓度随时间的变化趋势。")
    elif "dashboard" in fname:
        if hydro is not None and wq is not None:
            st.caption(f"综合仪表盘 — 汇总水位、流量、流速及水质浓度关键指标，一图纵览模拟全貌。")
        else:
            st.caption(f"综合仪表盘 — 汇总水动力和水质模拟关键结果。")
    else:
        st.caption(f"{title} — 基于雅瑶水道一维水动力-水质模型生成。")

# ---------- 图表工具函数 ----------

def tool_generate_time_series():
    df = st.session_state.df; black = st.session_state.black_spots
    if df is None: return "数据未加载"
    font = st.session_state.font_prop; fig, axes = plt.subplots(4,1,figsize=(12,10),sharex=True)
    axes[0].plot(df['datetime'], df['ammonia_nitrogen'], linewidth=0.8); axes[0].axhline(2.0,color='r',ls='--')
    if black is not None and len(black) > 0:
        sc_colors = black['severity_color'].values if 'severity_color' in black.columns else 'red'
        axes[0].scatter(black['datetime'], black['ammonia_nitrogen'], c=sc_colors, s=10, edgecolors='black', linewidths=0.2)
    axes[0].set_ylabel('氨氮 (mg/L)', fontproperties=font); axes[0].grid(True,alpha=0.3)
    axes[1].plot(df['datetime'], df['dissolved_oxygen'], linewidth=0.8); axes[1].axhline(2.0,color='r',ls='--')
    if black is not None and len(black) > 0:
        sc_colors = black['severity_color'].values if 'severity_color' in black.columns else 'red'
        axes[1].scatter(black['datetime'], black['dissolved_oxygen'], c=sc_colors, s=10, edgecolors='black', linewidths=0.2)
    axes[1].set_ylabel('溶解氧 (mg/L)', fontproperties=font); axes[1].grid(True,alpha=0.3)
    axes[2].plot(df['datetime'], df['turbidity'], linewidth=0.8)
    if black is not None and len(black) > 0:
        sc_colors = black['severity_color'].values if 'severity_color' in black.columns else 'red'
        axes[2].scatter(black['datetime'], black['turbidity'], c=sc_colors, s=10, edgecolors='black', linewidths=0.2)
    axes[2].set_ylabel('浑浊度 (NTU)', fontproperties=font); axes[2].grid(True,alpha=0.3)
    axes[3].plot(df['datetime'], df['ph'], linewidth=0.8, color='green'); axes[3].axhline(6.0,color='gray',ls='--'); axes[3].axhline(9.0,color='gray',ls='--')
    if black is not None and len(black) > 0:
        sc_colors = black['severity_color'].values if 'severity_color' in black.columns else 'red'
        axes[3].scatter(black['datetime'], black['ph'], c=sc_colors, s=10, edgecolors='black', linewidths=0.2)
    axes[3].set_ylabel('pH', fontproperties=font); axes[3].set_xlabel('时间', fontproperties=font); axes[3].grid(True,alpha=0.3)
    # 黑臭等级图例 (散点颜色来源: 《城市黑臭水体整治工作指南》)
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color='#FFA500', label='轻度黑臭'),
        Patch(color='#8B0000', label='重度黑臭'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=8, framealpha=0.8)
    plt.suptitle('水质时间序列与黑点分布', fontsize=14, fontproperties=font); plt.tight_layout()
    fig.savefig('time_series.png', dpi=150); plt.close(fig)
    st.session_state.charts_generated = True
    # 生成图表说明
    _build_time_series_caption(df, black)
    # 返回数据摘要供 LLM 消费 (非截断)
    black = st.session_state.black_spots
    summary = {"图表": "时间序列图已生成", "文件": "time_series.png"}
    if black is not None and len(black) > 0:
        summary["黑点概览"] = {
            "总黑点数": len(black),
            "黑臭分布": black['severity_label'].value_counts().to_dict(),
            "氨氮峰值_mgL": round(float(black['ammonia_nitrogen'].max()), 1),
            "DO最低_mgL": round(float(black['dissolved_oxygen'].min()), 2),
        }
    return json.dumps(summary, ensure_ascii=False)

def reverse_geocode(lat, lon, amap_key):
    url = "https://restapi.amap.com/v3/geocode/regeo"
    params = {"location":f"{lon},{lat}","key":amap_key,"radius":1000,"extensions":"all"}
    try:
        resp = requests.get(url, params=params, timeout=10); data = resp.json()
        if data.get("status")=="1":
            regeo = data["regeocode"]; address = regeo.get("formatted_address","")
            province = regeo.get("addressComponent",{}).get("province","")
            pois = [{"名称":p.get("name",""),"类型":p.get("type",""),"距离":p.get("distance","")} for p in (regeo.get("pois") or [])[:10]]
            return json.dumps({"地址":address,"省份":province,"周边POI":pois}, ensure_ascii=False)
        return "逆地理编码失败"
    except: return "逆地理编码异常"

async def _call_mcp(tool_name, params=None):
    client = MCPClient(); await client.start(); result = await client.call_tool(tool_name, params); await client.stop(); return result

def tool_get_national_station_data(province="", keyword=""):
    params = {};
    if province: params["province"] = province
    if keyword: params["search"] = keyword
    try:
        result = asyncio.run(_call_mcp("get_water_quality", params))
        if isinstance(result, dict) and "data" in result: return json.dumps(result["data"][:10], ensure_ascii=False)
        return str(result)
    except Exception as e: return f"获取国控站点数据失败: {e}"

def tool_get_satellite_image(lat, lon, zoom=16, cloud_token=None):
    token = cloud_token or _get_starcloud_token()
    if not token or "你的星图" in token:
        return "未配置星图地球 Token，请在侧边栏填写"
    # 将经纬度转为 XYZ 瓦片编号
    n = 2 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    tile_url = f"https://tiles1.geovisearth.com/base/v1/img/{zoom}/{x_tile}/{y_tile}?format=webp&tmsIds=w&token={token}"
    try:
        resp = requests.get(tile_url, timeout=15)
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
            return json.dumps({
                "状态": "卫星影像可用",
                "坐标": f"{lat:.6f}, {lon:.6f}",
                "缩放级别": zoom,
                "瓦片编号": f"z={zoom}, x={x_tile}, y={y_tile}",
                "影像大小": f"{len(resp.content)} 字节"
            }, ensure_ascii=False)
        else:
            return f"卫星影像获取失败: HTTP {resp.status_code}，缩放级别 {zoom} 可能超出覆盖范围"
    except Exception as e:
        return f"卫星影像请求异常: {e}"

def tool_generate_report(api_key, summary):
    client = OpenAI(base_url=BASE_URL, api_key=api_key)
    resp = client.chat.completions.create(model=MODEL_NAME, messages=[{"role":"user","content":f"你是水环境溯源专家。请严格依据GB 3838-2002撰写溯源报告。\n{summary}"}], temperature=0.1, max_tokens=2000)
    return resp.choices[0].message.content

# ---------- 工具注册 ----------
tools = [
    {"type":"function","function":{"name":"get_data_summary","description":"获取水质数据统计","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"find_black_spots","description":"基于《城市黑臭水体整治工作指南》(2015)和GB3838-2002发现水质黑点并DBSCAN聚类，返回黑臭等级分布(轻度/重度)和聚类详情","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"analyze_video","description":"分析视频帧","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"cross_validate_video","description":"将视频帧视觉特征(Loaclacian方差/HSV)与走航监测数据(浊度/氨氮/DO/COD)交叉印证,计算Pearson相关系数和黑臭判定一致率","parameters":{"type":"object","properties":{"time_offset_sec":{"type":"integer","description":"视频第0秒相对第一条记录的时间偏移,默认0"},"time_window_sec":{"type":"integer","description":"匹配窗口宽度秒,默认30"}}}}},
    {"type":"function","function":{"name":"generate_cross_validation_chart","description":"生成视频-传感器交叉印证双面板对比图","parameters":{"type":"object","properties":{"time_offset_sec":{"type":"integer","description":"时间偏移秒,默认0"}}}}},
    {"type":"function","function":{"name":"get_comprehensive_summary","description":"获取综合分析摘要JSON,聚合数据概览/黑臭检测(聚类详情+Top5严重点位)/空间分析/交叉印证/河道模型所有结果","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"generate_final_report","description":"生成溯源报告","parameters":{"type":"object","properties":{"summary":{"type":"string"}},"required":["summary"]}}},
    {"type":"function","function":{"name":"generate_time_series_chart","description":"绘制时间序列图","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"generate_trace_map","description":"生成交互地图","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"reverse_geocode","description":"将精确经纬度转为地址/省份","parameters":{"type":"object","properties":{"lat":{"type":"number"},"lon":{"type":"number"}},"required":["lat","lon"]}}},
    {"type":"function","function":{"name":"search_nearby_pollution_sources","description":"搜索附近污染源（按距离排序）","parameters":{"type":"object","properties":{"lat":{"type":"number"},"lon":{"type":"number"},"radius":{"type":"number"}},"required":["lat","lon"]}}},
    {"type":"function","function":{"name":"get_national_station_data","description":"获取国控站点数据","parameters":{"type":"object","properties":{"province":{"type":"string"},"keyword":{"type":"string"}}}}},
    {"type":"function","function":{"name":"get_satellite_image","description":"获取卫星影像","parameters":{"type":"object","properties":{"lat":{"type":"number"},"lon":{"type":"number"},"zoom":{"type":"integer","default":16}},"required":["lat","lon"]}}},
    {"type":"function","function":{"name":"init_river_model","description":"初始化雅瑶水道一维水动力-水质模型。加载河道几何断面(默认50个)、曼宁糙率、潮汐边界条件。返回河道参数摘要。","parameters":{"type":"object","properties":{"n_cross_sections":{"type":"integer","description":"断面数量(30-100,默认50)"},"mannings_n":{"type":"number","description":"曼宁糙率系数(0.025-0.035,默认0.028)"}}}}},
    {"type":"function","function":{"name":"run_hydrodynamic_simulation","description":"运行雅瑶水道一维水动力模拟(Saint-Venant方程)。返回各断面水位、流速、流量统计。必须先调用init_river_model。","parameters":{"type":"object","properties":{"upstream_flow_m3s":{"type":"number","description":"上游平均流量m³/s(默认6.9)"},"downstream_stage_m":{"type":"number","description":"下游平均潮位m(默认0.83)"},"tidal_amplitude_m":{"type":"number","description":"潮汐振幅m(默认0.8,半潮差)"},"tidal_period_h":{"type":"number","description":"潮汐周期h(默认12.42,M2分潮)"},"duration_h":{"type":"number","description":"模拟时长h(默认24)"},"dt_s":{"type":"number","description":"计算时间步长s(默认60)"}}}}},
    {"type":"function","function":{"name":"simulate_pollution_event","description":"在雅瑶水道指定桩号模拟污染事件,运行ADR(对流-扩散-反应)水质模型。模拟污染物输移扩散和生化反应。返回峰值浓度、影响范围、达标时间。必须先运行水动力模拟。","parameters":{"type":"object","properties":{"chainage_m":{"type":"number","description":"污染源桩号m,从上游起算(0-3200,默认1600)"},"pollutant_type":{"type":"string","description":"污染物类型:ammonia(氨氮)/cbod(碳BOD)/cod(化学需氧量)/conservative(保守示踪剂)"},"load_kg":{"type":"number","description":"污染物总质量kg(默认50)"},"duration_min":{"type":"number","description":"排放持续时间min(默认30)"},"simulation_hours":{"type":"number","description":"模拟总时长h(默认24)"}}}}},
    {"type":"function","function":{"name":"get_concentration_profile","description":"查询指定时刻的沿程浓度剖面。返回各断面浓度、超标断面位置、最大浓度位置。","parameters":{"type":"object","properties":{"time_h":{"type":"number","description":"查询时刻h(默认6)"},"constituent":{"type":"string","description":"查询组分:ammonia/cbod/do(默认ammonia)"}}}}},
    {"type":"function","function":{"name":"compare_scenarios","description":"对比两个已保存的模拟情景。返回对比表格(峰值浓度/影响范围/达标时间差异)。","parameters":{"type":"object","properties":{"scenario1_name":{"type":"string","description":"情景1名称(留空则自动选择最新)"},"scenario2_name":{"type":"string","description":"情景2名称(留空则自动选择次新)"}}}}},
    {"type":"function","function":{"name":"export_river_model_data","description":"导出模拟结果数据为JSON或CSV文件。","parameters":{"type":"object","properties":{"format":{"type":"string","description":"导出格式:json或csv(默认json)"},"scenario_name":{"type":"string","description":"情景名称(默认latest)"}}}}},
]

MODEL_NAME = "deepseek-chat"
BASE_URL = "https://api.deepseek.com/v1"

def execute_tool(name, args, api_key, amap_key, cloud_token=None):
    if name == "get_data_summary": return tool_data_summary()
    elif name == "find_black_spots": return tool_find_black_spots()
    elif name == "analyze_video": return tool_video_analysis()
    elif name == "cross_validate_video": return tool_cross_validate_video(**args)
    elif name == "generate_cross_validation_chart": return tool_generate_cross_validation_chart(**args)
    elif name == "get_comprehensive_summary": return tool_get_comprehensive_summary()
    elif name == "generate_final_report": return tool_generate_report(api_key, **args)
    elif name == "generate_time_series_chart": return tool_generate_time_series()
    elif name == "generate_trace_map": return tool_generate_trace_map(amap_key, cloud_token)
    elif name == "reverse_geocode": return reverse_geocode(args.get("lat"), args.get("lon"), amap_key)
    elif name == "search_nearby_pollution_sources":
        black = st.session_state.black_spots
        if black is not None and len(black)>0:
            best_cluster_id = black['cluster_id'].value_counts().idxmax()
            best_cluster = black[black['cluster_id']==best_cluster_id]
            lat = best_cluster['latitude'].mean()
            lon = best_cluster['longitude'].mean()
        else:
            lat = args.get("lat"); lon = args.get("lon")
        pois = search_nearby_pollution_sources(lat, lon, amap_key, args.get("radius",500))
        return json.dumps({"搜索中心坐标":f"{lat:.6f},{lon:.6f}","最近污染源（按距离排序）":pois[:15]}, ensure_ascii=False)
    elif name == "get_national_station_data":
        return tool_get_national_station_data(args.get("province",""), args.get("keyword",""))
    elif name == "get_satellite_image":
        return tool_get_satellite_image(args.get("lat"), args.get("lon"), args.get("zoom",16), cloud_token)
    elif name == "init_river_model": return tool_river_init(st.session_state.river_state, **args)
    elif name == "run_hydrodynamic_simulation": return tool_river_hydro_simulation(st.session_state.river_state, **args)
    elif name == "simulate_pollution_event": return tool_river_pollution_event(st.session_state.river_state, **args)
    elif name == "get_concentration_profile": return tool_river_concentration_profile(st.session_state.river_state, **args)
    elif name == "compare_scenarios": return tool_river_compare_scenarios(st.session_state.river_state, **args)
    elif name == "export_river_model_data": return tool_river_export_data(st.session_state.river_state, **args)
    else: return f"未知工具: {name}"

def _ensure_charts(amap_key, cloud_token=None):
    if not os.path.exists("time_series.png") and st.session_state.df is not None:
        tool_generate_time_series()
    if not os.path.exists("trace_map.html") and st.session_state.df is not None:
        tool_generate_trace_map(amap_key, cloud_token)
    # 河道模型图表
    if st.session_state.river_state.has_hydro() and not os.path.exists("river_hydro_profile.png"):
        from river_model.visualization import plot_water_level_profile, plot_discharge_profile
        try:
            plot_water_level_profile(st.session_state.river_state.hydro_result,
                                     st.session_state.river_state.config, "river_hydro_profile.png")
            plot_discharge_profile(st.session_state.river_state.hydro_result, "river_discharge_profile.png")
        except Exception:
            pass

def _trim_tool_messages(msgs):
    """截断过长的工具返回消息，防止上下文爆炸"""
    trimmed = []
    for m in msgs:
        # 兼容 dict 和 ChatCompletionMessage 两种类型
        if isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")
        else:
            role = getattr(m, "role", "")
            content = getattr(m, "content", "") or ""
        if role == "tool" and len(content) > 2000:
            if isinstance(m, dict):
                m = m.copy()
                m["content"] = content[:2000] + "\n...(已截断)"
            else:
                m = m.model_copy()
                m.content = content[:2000] + "\n...(已截断)"
        trimmed.append(m)
    return trimmed

def process_user_message(user_text, api_key, amap_key, cloud_token=None, system_prompt=None):
    if system_prompt is None:
        system_prompt = "你是水质监测与污染溯源专家。所有推断必须基于监测数据和GB3838标准。禁止编造坐标。"
    msgs = st.session_state.full_messages.copy()
    if len(msgs)==0 or msgs[0]["role"]!="system": msgs.insert(0,{"role":"system","content":system_prompt})
    msgs.append({"role":"user","content":user_text})
    client = OpenAI(base_url=BASE_URL, api_key=api_key)
    max_turns = 15
    turn = 0
    while turn < max_turns:
        turn += 1
        resp = client.chat.completions.create(model=MODEL_NAME, messages=msgs, tools=tools, tool_choice="auto", temperature=0.1)
        msg = resp.choices[0].message; msgs.append(msg)
        if not msg.tool_calls:
            msgs = _trim_tool_messages(msgs)
            st.session_state.full_messages = msgs
            _ensure_charts(amap_key, cloud_token)
            return msg.content or ""
        for tc in msg.tool_calls:
            raw = json.loads(tc.function.arguments) if tc.function.arguments else {}
            res = execute_tool(tc.function.name, raw, api_key, amap_key, cloud_token)
            # 截断过长的工具返回，防止上下文爆炸 (综合分析摘要放宽到4000)
            limit = 4000 if tc.function.name == "get_comprehensive_summary" else 2000
            if len(res) > limit:
                res = res[:limit] + "\n...(已截断)"
            msgs.append({"role":"tool","tool_call_id":tc.id,"content":res})
    # 兜底：超过最大轮次后强制结束
    msgs.append({"role":"user","content":"已达到最大分析轮次。请基于已完成的分析给出最终结论，禁止再调用工具。"})
    final = client.chat.completions.create(model=MODEL_NAME, messages=msgs, temperature=0.1)
    msgs.append(final.choices[0].message)
    msgs = _trim_tool_messages(msgs)
    st.session_state.full_messages = msgs
    _ensure_charts(amap_key, cloud_token)
    return final.choices[0].message.content or ""

# ---------- UI ----------
st.title("🌊 水质监测与污染溯源助手")

# 首次运行配置向导：未配置 API key 时阻塞整个页面
if not _show_setup_wizard():
    st.stop()

with st.sidebar:
    st.header("📁 数据上传与配置")
    api_key = st.text_input("DeepSeek API Key", type="password",
                           value=st.session_state.get("_deepseek_key", ""))
    amap_key = st.text_input("高德地图 API Key (可选)", type="password",
                            value=st.session_state.get("_amap_key", ""),
                            help="用于底图/逆地理/POI搜索")
    st_cloud_token = st.text_input("星图地球 Token (可选)", type="password",
                                  value=st.session_state.get("_starcloud_token", ""),
                                  help="https://datacloud.geovisearth.com 注册")
    uploaded_file = st.file_uploader("上传走航数据 (xlsx/xls/csv)", type=["xlsx","xls","csv"])
    uploaded_video = st.file_uploader("上传样本视频 (可选)", type=["mp4","avi","mov"])
    if uploaded_file and st.button("加载数据"):
        with st.spinner("加载数据..."): msg = load_data(uploaded_file); st.success(msg)
        if uploaded_video:
            with st.spinner("提取视频帧..."): msg = extract_video(uploaded_video, interval=10); st.success(msg)
    if st.session_state.df is not None:
        st.write(f"✅ 数据：{len(st.session_state.df)} 条"); st.write(f"🎥 视频帧：{len(st.session_state.video_frames)} 帧")
        if len(st.session_state.video_frames) > 0:
            col_cv1, col_cv2 = st.columns(2)
            with col_cv1:
                time_offset = st.number_input("时间偏移 (秒)", value=0, step=10, help="视频第0秒相对第一条监测记录的偏移, 正=视频滞后")
            with col_cv2:
                if st.button("🔬 交叉印证", use_container_width=True):
                    with st.spinner("视频-传感器交叉印证中..."):
                        cv_result = tool_cross_validate_video(time_offset_sec=time_offset)
                        st.session_state.cross_validation_result = json.loads(cv_result)
                        from water_quality_mcp.src.water_quality_mcp.charts import generate_cross_validation_chart, generate_correlation_heatmap
                        from water_quality_mcp.src.water_quality_mcp.state import WaterQualityState
                        ws = WaterQualityState()
                        ws.df = st.session_state.df
                        ws.video_frames = st.session_state.video_frames
                        ws.black_spots = st.session_state.black_spots
                        generate_cross_validation_chart(ws, "cross_validation.png", time_offset)
                        generate_correlation_heatmap(st.session_state.cross_validation_result, "correlation_heatmap.png")
                        _build_cross_validation_caption()
                        _build_correlation_heatmap_caption()
                    st.success("交叉印证完成，查看「分析图表」标签")
                    st.rerun()

tab1, tab2, tab3 = st.tabs(["💬 智能对话", "📊 分析图表", "🌊 河道模型"])
with tab1:
    for m in st.session_state.messages:
        if m["role"]=="user": st.chat_message("user").markdown(m["content"])
        elif m["role"]=="assistant": st.chat_message("assistant").markdown(m["content"])
    if st.session_state.df is not None and not st.session_state.analysis_done:
        if st.button("🚀 开始自动分析", type="primary"):
            if not api_key: st.error("请输入API Key")
            else:
                system_prompt = (
                    "你是自动分析Agent。严格按以下顺序执行，不得跳过：\n"
                    "1. get_data_summary\n"
                    "2. find_black_spots\n"
                    "3. reverse_geocode（使用最密集聚类的中心坐标）\n"
                    "4. get_national_station_data\n"
                    "5. analyze_video\n"
                    "5.5. cross_validate_video（视频与传感器交叉印证）\n"
                    "6. search_nearby_pollution_sources\n"
                    "7. get_satellite_image\n"
                    "8. generate_time_series_chart（必须完成）\n"
                    "9. generate_trace_map（必须完成）\n"
                    "10. get_comprehensive_summary（获取聚合所有分析结果的统一JSON）\n"
                    "11. generate_final_report（基于 comprehensive_summary 编写溯源报告）\n\n"
                    "输出要求：\n"
                    "- 为每张生成的图表撰写一句简要说明（时间序列图、溯源地图、交叉印证图），说明图表展示的核心发现。\n"
                    "- 只输出极简结论（不超过五句话），并提示用户查看「分析图表」标签。\n"
                    "- 严禁输出任何思考过程、自我检讨、道歉或工具调用日志。\n"
                    "- 严禁输出 DSML 标签。"
                )
                with st.spinner("Agent 分析中..."):
                    final = process_user_message("请开始自动分析。", api_key, amap_key, st_cloud_token, system_prompt)
                    st.session_state.messages.append({"role":"assistant","content":final})
                    st.session_state.analysis_done = True; st.rerun()
    user_input = st.chat_input("请输入问题...")
    if user_input:
        if not api_key: st.error("请输入API Key")
        elif st.session_state.df is None: st.error("请先上传并加载数据")
        else:
            st.session_state.messages.append({"role":"user","content":user_input}); st.chat_message("user").markdown(user_input)
            with st.spinner("思考中..."):
                answer = process_user_message(user_input, api_key, amap_key, st_cloud_token)
                st.session_state.messages.append({"role":"assistant","content":answer}); st.rerun()

with tab2:
    st.header("📈 分析图表")
    captions = st.session_state.chart_captions
    if os.path.exists("time_series.png"):
        st.subheader("时间序列趋势")
        st.image("time_series.png")
        if captions.get("time_series"):
            st.caption(captions["time_series"])
        with open("time_series.png", "rb") as f:
            st.download_button("下载时间序列图", f, file_name="time_series.png")
    else:
        st.info("尚未生成时间序列图")
    if os.path.exists("trace_map.html"):
        st.subheader("交互式溯源地图")
        if captions.get("trace_map"):
            st.caption(captions["trace_map"])
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("🔄 刷新地图", key="refresh_map"):
                if st.session_state.df is not None and st.session_state.black_spots is not None:
                    with st.spinner("重新生成地图..."):
                        tool_generate_trace_map(amap_key, st_cloud_token)
                    st.rerun()
        with open("trace_map.html", "r", encoding="utf-8") as f:
            html = f.read()
        st.iframe("trace_map.html", height=600)
        st.download_button("下载交互式地图", html, file_name="trace_map.html", mime="text/html")
    else:
        st.info("尚未生成溯源地图")
    # 交叉印证图表
    st.subheader("视频-传感器交叉印证")
    if os.path.exists("cross_validation.png"):
        st.image("cross_validation.png")
        if captions.get("cross_validation"):
            st.caption(captions["cross_validation"])
        with open("cross_validation.png", "rb") as f:
            st.download_button("下载交叉印证图", f, file_name="cross_validation.png")
    elif len(st.session_state.video_frames) > 0:
        st.info("视频帧已提取，点击按钮运行交叉印证")
        cv_offset = st.number_input("时间偏移 (秒)", value=0, step=10, key="cv_offset_tab2")
        if st.button("🔬 运行交叉印证", key="cv_btn_tab2"):
            with st.spinner("交叉印证中..."):
                cv_result = tool_cross_validate_video(time_offset_sec=cv_offset)
                st.session_state.cross_validation_result = json.loads(cv_result)
                from water_quality_mcp.src.water_quality_mcp.charts import generate_cross_validation_chart as _cv_chart, generate_correlation_heatmap as _cv_heat
                from water_quality_mcp.src.water_quality_mcp.state import WaterQualityState
                ws = WaterQualityState()
                ws.df = st.session_state.df
                ws.video_frames = st.session_state.video_frames
                ws.black_spots = st.session_state.black_spots
                _cv_chart(ws, "cross_validation.png", cv_offset)
                _cv_heat(st.session_state.cross_validation_result, "correlation_heatmap.png")
                _build_cross_validation_caption()
                _build_correlation_heatmap_caption()
            st.success("交叉印证完成")
            st.rerun()
    else:
        st.info("尚未生成交叉印证图 (需先上传视频并运行交叉印证)")
    if os.path.exists("correlation_heatmap.png"):
        st.image("correlation_heatmap.png")
        if captions.get("correlation_heatmap"):
            st.caption(captions["correlation_heatmap"])
        with open("correlation_heatmap.png", "rb") as f:
            st.download_button("下载相关系数热力图", f, file_name="correlation_heatmap.png")

with tab3:
    st.header("🌊 雅瑶水道 水动力-水质模型")
    st.markdown("基于 Preissmann 隐式格式求解 Saint-Venant 方程 + ADR 水质模型。模拟潮汐影响下污染物输移扩散和生化反应。")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        n_cs = st.slider("断面数", 20, 100, 50, 10, key="river_ncs")
        n_val = st.slider("曼宁糙率", 0.020, 0.040, 0.028, 0.001, key="river_n")
    with col_r2:
        flow_val = st.slider("上游流量 (m³/s)", 1.0, 150.0, 6.9, 1.0, key="river_flow")
        tide_amp = st.slider("潮汐振幅 (m)", 0.0, 1.5, 0.8, 0.05, key="river_tide")

    if st.button("🚀 运行雅瑶水道模型", type="primary", use_container_width=True):
        if not api_key:
            st.error("请先在侧边栏输入 DeepSeek API Key")
        else:
            with st.spinner("模型计算中..."):
                init_msg = tool_river_init(st.session_state.river_state, n_cross_sections=n_cs, mannings_n=n_val)
                st.success("模型初始化完成")

                hydro_msg = tool_river_hydro_simulation(
                    st.session_state.river_state,
                    upstream_flow_m3s=flow_val,
                    tidal_amplitude_m=tide_amp,
                    duration_h=24.0,
                )
                st.success("水动力模拟完成")

                poll_msg = tool_river_pollution_event(
                    st.session_state.river_state,
                    chainage_m=1600.0,
                    pollutant_type="ammonia",
                    load_kg=50.0,
                    duration_min=30.0,
                    simulation_hours=24.0,
                )
                st.success("污染事件模拟完成")

                st.markdown("### 模拟结果摘要")
                st.json(json.loads(hydro_msg))
                st.json(json.loads(poll_msg))

    # ---- 溯源动画生成 ----
    st.markdown("---")
    st.markdown("### 🔍 污染溯源动画")
    st.caption("基于实测数据定位污染源工厂 → 非恒定流模拟 → 生成扩散动画")

    col_anim1, col_anim2 = st.columns(2)
    with col_anim1:
        anim_source_chainage = st.number_input(
            "污染源桩号 (m, 0=自动)", value=0.0, step=100.0, key="anim_src_ch"
        )
    with col_anim2:
        anim_load = st.number_input(
            "排放量 (kg, 0=自动估算)", value=0.0, step=50.0, key="anim_load"
        )

    if st.button("🎬 生成污染溯源动画", type="secondary", use_container_width=True):
        if not os.path.exists("_uploaded_data.xlsx"):
            st.warning("请先在侧边栏上传并加载走航数据，再生成溯源动画")
        else:
            with st.spinner("正在生成溯源动画 (约需 30 秒)..."):
                from build_animation import build_trace_animation

                def progress(msg):
                    pass  # Streamlit 的 spinner 已提供视觉反馈

                try:
                    out_path = build_trace_animation(
                        excel_path="_uploaded_data.xlsx",
                        output_path="river_pollution_animation.html",
                        progress_callback=progress,
                    )
                    # 缓存到 session_state, 避免切换标签后丢失
                    if os.path.exists(out_path):
                        with open(out_path, "r", encoding="utf-8") as f:
                            st.session_state.anim_html = f.read()
                        st.session_state.anim_file_size = os.path.getsize(out_path)
                    _build_river_animation_caption(out_path)
                    st.success(f"动画已生成: {out_path}")
                except Exception as e:
                    st.error(f"生成失败: {e}")

    # 显示已生成的动画
    anim_file = "river_pollution_animation.html"
    if os.path.exists(anim_file):
        st.markdown("---")
        captions = st.session_state.chart_captions
        if captions.get("river_animation"):
            st.caption(captions["river_animation"])
        st.caption("💡 切换标签后动画会自动恢复到上次播放位置")
        # 优先使用缓存的 HTML, 文件变更时自动刷新
        current_size = os.path.getsize(anim_file)
        if st.session_state.anim_html is None or current_size != st.session_state.anim_file_size:
            with open(anim_file, "r", encoding="utf-8") as f:
                st.session_state.anim_html = f.read()
            st.session_state.anim_file_size = current_size
        st.iframe(anim_file, height=700)
        st.caption(f"文件: {anim_file} ({current_size/1024:.0f} KB)")
        with open(anim_file, "rb") as f:
            st.download_button("📥 下载动画", f, file_name=anim_file, mime="text/html")

    # 显示已生成的河道模型图表
    river_files = [
        ("river_hydro_profile.png", "沿程水位剖面"),
        ("river_discharge_profile.png", "沿程流量剖面"),
        ("river_wq_contour.png", "浓度时空分布"),
        ("river_wq_profile.png", "浓度沿程剖面"),
        ("river_wq_timeseries.png", "浓度过程线"),
        ("river_dashboard.png", "综合仪表盘"),
    ]
    existing = [(f, t) for f, t in river_files if os.path.exists(f)]
    if existing:
        st.markdown("---")
        st.markdown("### 已生成的河道模型图表")
        rs = st.session_state.river_state
        for fname, title in existing:
            st.markdown(f"**{title}**")
            st.image(fname)
            _render_river_chart_caption(fname, title, rs)

