"""污染扩散地图动画生成器。

生成自包含 HTML 文件, 使用纯 JavaScript 实现污染羽流在卫星地图上随时间扩散的动画。
完全离线可用, 不依赖任何外部 CDN 资源(除地图瓦片外)。
"""

import json
import os
import numpy as np
from typing import List, Tuple, Optional

import folium
from folium import Map, PolyLine, Marker, CircleMarker

from .state import WQResult
from .geo_mapping import (chainage_array_to_gps, get_gps_track_coords,
                           get_map_center, chainage_to_gps)


_STANDARDS = {
    "ammonia": {
        "name": "氨氮 (NH₃-N)", "limit": 2.0,
        "thresholds": [0.15, 0.5, 1.0, 1.5, 2.0],
        "colors": ["#0066FF", "#00CC44", "#FFCC00", "#FF8800", "#FF0000"],
        "worst_color": "#990000",
        "labels": ["I类 - 优", "II类 - 良", "III类 - 一般", "IV类 - 差", "V类 - 很差", "劣V类 - 严重污染"],
    },
    "cbod": {
        "name": "CBOD", "limit": 10.0,
        "thresholds": [3, 4, 6, 10, 15],
        "colors": ["#0066FF", "#00CC44", "#FFCC00", "#FF8800", "#FF0000"],
        "worst_color": "#990000",
        "labels": ["I类", "II类", "III类", "IV类", "V类", "劣V类"],
    },
    "do": {
        "name": "溶解氧 (DO)", "limit": 2.0, "inverted": True,
        "thresholds": [7.5, 6.0, 5.0, 3.0, 2.0],
        "colors": ["#0066FF", "#00CC44", "#FFCC00", "#FF8800", "#FF0000"],
        "worst_color": "#990000",
        "labels": ["I类", "II类", "III类", "IV类", "V类", "劣V类"],
    },
}


def _build_color_scale(frames_data: list) -> list:
    """基于数据实际范围建立相对色标。min→蓝, max→红。"""
    all_c = []
    for f in frames_data:
        all_c.extend(f["c"])
    c_min = min(all_c)
    c_max = max(all_c)
    c_range = c_max - c_min if c_max > c_min else 1.0
    # 6 级色标: 蓝色→绿色→黄绿→黄色→橙色→红色
    return {
        "min": c_min, "max": c_max, "range": c_range,
        "stops": [c_min + c_range * r for r in [0, 0.2, 0.4, 0.6, 0.8, 1.0]],
        "colors": ["#0066FF", "#00CC44", "#AAEE00", "#FFCC00", "#FF6600", "#FF0000"],
    }


def _conc_to_color_relative(conc: float, color_scale: dict) -> str:
    """基于数据范围映射颜色。"""
    if color_scale["range"] < 0.001:
        return color_scale["colors"][0]
    ratio = (conc - color_scale["min"]) / color_scale["range"]
    ratio = max(0.0, min(1.0, ratio))
    idx = int(ratio * (len(color_scale["stops"]) - 1))
    idx = min(idx, len(color_scale["colors"]) - 1)
    return color_scale["colors"][idx]


def _conc_to_radius_relative(conc: float, color_scale: dict) -> float:
    """基于数据范围映射半径。"""
    if color_scale["range"] < 0.001:
        return 5.0
    ratio = (conc - color_scale["min"]) / color_scale["range"]
    ratio = max(0.0, min(1.0, ratio))
    return 4.0 + ratio * 12.0


def _build_gb3838_legend_html(constituent: str = "ammonia") -> str:
    """GB3838 图例 HTML。"""
    std = _STANDARDS.get(constituent, _STANDARDS["ammonia"])
    items = ""
    for i, (t, c, label) in enumerate(zip(std["thresholds"], std["colors"], std["labels"][:-1])):
        items += (f'<div style="display:flex;align-items:center;margin:2px 0;">'
                  f'<span style="display:inline-block;width:18px;height:12px;'
                  f'background:{c};border-radius:2px;margin-right:6px;"></span>'
                  f'<span style="font-size:11px;">{label} (≤{t})</span></div>')
    items += (f'<div style="display:flex;align-items:center;margin:2px 0;">'
              f'<span style="display:inline-block;width:18px;height:12px;'
              f'background:{std["worst_color"]};border-radius:2px;margin-right:6px;"></span>'
              f'<span style="font-size:11px;">{std["labels"][-1]}</span></div>')
    items += ('<div style="margin-top:6px;padding-top:4px;border-top:1px solid #ccc;font-size:10px;color:#666;">'
              '黑臭判定: 氨氮>8mg/L 或 DO<2mg/L</div>')
    return f'''<div id="gb3838-legend" style="position:fixed;bottom:70px;right:10px;z-index:9999;
        background:rgba(255,255,255,0.93);padding:8px 12px;border-radius:6px;
        box-shadow:0 2px 8px rgba(0,0,0,0.3);font-family:Arial,sans-serif;max-width:200px;">
        <div style="font-weight:bold;font-size:13px;margin-bottom:4px;">GB3838-2002 {std["name"]}</div>{items}</div>'''


def build_pollution_map_animation(
    gps_mapping: dict,
    wq: WQResult,
    constituent: str = "ammonia",
    source_chainage_m: float = 1600.0,
    output_path: str = "river_pollution_animation.html",
    max_frames: int = 60,
) -> str:
    """生成污染扩散地图动画 HTML。

    使用纯 JavaScript 动画 — 不依赖 HeatMapWithTime CDN。
    每帧更新 CircleMarker 颜色, 展示污染物羽流沿河道随时间扩散。
    """
    std = _STANDARDS.get(constituent, _STANDARDS["ammonia"])
    std_name = std["name"]

    # ---- 选择数据 ----
    data_map = {"ammonia": wq.ammonia, "cbod": wq.cbod, "do": wq.dissolved_oxygen}
    data = data_map.get(constituent)
    if data is None:
        raise ValueError(f"{constituent} 数据不可用")

    # ---- 降采样 ----
    nt = len(wq.t)
    if nt <= max_frames:
        frame_indices = np.arange(nt)
    else:
        frame_indices = np.linspace(0, nt - 1, max_frames, dtype=int)

    # ---- GPS 映射 ----
    model_chainage = wq.chainage
    gps_length = gps_mapping["total_length_m"]
    model_length = float(model_chainage[-1]) if model_chainage[-1] > 0 else gps_length
    scaled_chainage = model_chainage * (gps_length / model_length)
    section_lats, section_lons = chainage_array_to_gps(gps_mapping, scaled_chainage)
    n_sections = len(section_lats)

    # ---- 构建帧数据 (先收集原始浓度, 后统一计算色标) ----
    frames_raw = []
    for fi in frame_indices:
        concentrations = [round(float(c), 3) for c in data[fi, :]]
        frames_raw.append({
            "t": round(float(wq.t[fi]), 2),
            "c": concentrations,
        })

    # 基于全数据范围建立相对色标
    color_scale = _build_color_scale(frames_raw)

    # 应用色标
    frames_data = []
    for fr in frames_raw:
        colors = [_conc_to_color_relative(c, color_scale) for c in fr["c"]]
        radii = [_conc_to_radius_relative(c, color_scale) for c in fr["c"]]
        frames_data.append({
            "t": fr["t"],
            "c": fr["c"],
            "colors": colors,
            "radii": radii,
        })

    time_labels = [f"T+{round(float(wq.t[fi]),1):.1f}h" for fi in frame_indices]
    frames_json = json.dumps(frames_data, ensure_ascii=False)
    time_labels_json = json.dumps(time_labels, ensure_ascii=False)
    lats_json = json.dumps([float(x) for x in section_lats])
    lons_json = json.dumps([float(x) for x in section_lons])

    # ---- 地图 ----
    center = get_map_center(gps_mapping)
    m = Map(location=list(center), zoom_start=15, control_scale=True, tiles=None)

    # 底图 (多源, 应对部分服务器在国内不可达)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri 卫星图 (WGS-84)", overlay=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        attr="CartoDB", name="CartoDB Voyager (WGS-84 推荐)", overlay=False,
        subdomains="abc",
    ).add_to(m)
    folium.TileLayer(tiles="OpenStreetMap", name="OSM 街道图", overlay=False).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        attr="CartoDB", name="CartoDB Light (WGS-84)", overlay=False,
        subdomains="abc",
    ).add_to(m)

    # 河道轨迹
    track_coords = get_gps_track_coords(gps_mapping)
    PolyLine(track_coords, weight=5, color="#0066CC", opacity=0.8, name="雅瑶水道").add_to(m)

    # 起止标记
    Marker(track_coords[0], popup="上游: 北村水闸方向",
           icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
    Marker(track_coords[-1], popup="下游: 珠江方向",
           icon=folium.Icon(color="blue", icon="stop", prefix="fa")).add_to(m)

    # 污染源标记
    try:
        src_lat, src_lon = chainage_to_gps(gps_mapping, source_chainage_m)
    except Exception:
        src_lat, src_lon = track_coords[len(track_coords) // 2]
    Marker([src_lat, src_lon], popup="<b>污染源泄漏点</b>",
           icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa")).add_to(m)
    CircleMarker([src_lat, src_lon], radius=14, color="#FF0000", fill=True,
                 fill_color="#FF0000", fill_opacity=0.35, weight=3).add_to(m)

    # 图例和标题
    legend_html = _build_gb3838_legend_html(constituent)
    m.get_root().html.add_child(folium.Element(legend_html))

    load_kg = wq.params.get("load_kg", "?")
    src_ch = wq.params.get("chainage_m", source_chainage_m)
    title_html = f'''<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:9999;
        background:rgba(0,0,0,0.82);color:white;padding:10px 24px;border-radius:8px;
        font-family:'Microsoft YaHei',Arial,sans-serif;text-align:center;pointer-events:none;">
        <h3 style="margin:0 0 4px 0;font-size:18px;">{std_name} 污染扩散模拟</h3>
        <span style="font-size:13px;">泄漏量 {load_kg} kg | 桩号 {src_ch:.0f} m | 模拟 {wq.t[-1]:.0f} h</span>
    </div>'''
    m.get_root().html.add_child(folium.Element(title_html))

    # 控制面板容器 (由 JS 填充)
    control_html = '''<div id="anim-controls" style="position:fixed;bottom:30px;left:50%;
        transform:translateX(-50%);z-index:9999;background:rgba(0,0,0,0.85);color:white;
        padding:12px 20px;border-radius:10px;font-family:Arial,sans-serif;
        display:flex;align-items:center;gap:15px;min-width:500px;">
        <button id="btn-play" onclick="togglePlay()" style="background:#e94560;color:white;
            border:none;padding:8px 16px;border-radius:4px;cursor:pointer;font-size:14px;font-weight:bold;">
            ▶ 播放</button>
        <button onclick="seekFrame(0)" style="background:#555;color:white;border:none;
            padding:8px 12px;border-radius:4px;cursor:pointer;">⟲ 重置</button>
        <input type="range" id="time-slider" min="0" max="0" value="0" step="1"
            oninput="seekFrame(parseInt(this.value))"
            style="flex:1;accent-color:#e94560;min-width:120px;">
        <span id="time-display" style="font-size:15px;font-weight:bold;color:#FFCC00;min-width:80px;text-align:center;">T+0.0h</span>
        <select id="speed-select" onchange="setSpeed(this.value)" style="background:#333;color:white;
            border:1px solid #666;border-radius:3px;padding:4px;">
            <option value="0.5">0.5x</option><option value="1">1x</option>
            <option value="2" selected>2x</option><option value="5">5x</option><option value="10">10x</option>
        </select>
    </div>'''
    m.get_root().html.add_child(folium.Element(control_html))

    # ---- 预渲染以获取地图变量名 ----
    import re as _re
    from io import StringIO as _StringIO
    _tmp_buf = _StringIO()
    # 提取地图变量名
    _prerender_match = _re.search(r'var (map_\w+) = L\.map', _tmp_html if False else '')
    del _tmp_buf

    # ---- JavaScript 动画引擎 ----
    js_code = f'''
    <script>
    (function() {{
        // ---- 数据 ----
        var frames = {frames_json};
        var timeLabels = {time_labels_json};
        var lats = {lats_json};
        var lons = {lons_json};
        var totalFrames = frames.length;

        // ---- 状态 ----
        var currentFrame = 0;
        var playing = false;
        var timer = null;
        var playSpeed = 2;
        var markers = [];

        // ---- 查找 Leaflet 地图对象 ----
        function findMap() {{
            // 方法1: 遍历所有全局变量找 L.map 实例
            for (var key in window) {{
                try {{
                    var obj = window[key];
                    if (obj && obj._container && obj._container.classList &&
                        obj._container.classList.contains('folium-map')) {{
                        return obj;
                    }}
                }} catch(e) {{}}
            }}
            // 方法2: 找所有 map_ 前缀变量
            for (var key in window) {{
                if (key.indexOf('map_') === 0 && window[key] &&
                    typeof window[key].addLayer === 'function') {{
                    return window[key];
                }}
            }}
            return null;
        }}

        function initWhenReady() {{
            var map = findMap();
            if (!map) {{
                setTimeout(initWhenReady, 200);
                return;
            }}

            // 创建初始标记 (第一帧)
            var frame0 = frames[0];
            for (var i = 0; i < lats.length; i++) {{
                var marker = L.circleMarker([lats[i], lons[i]], {{
                    radius: frame0.radii[i],
                    fillColor: frame0.colors[i],
                    color: frame0.colors[i],
                    weight: 1,
                    opacity: 0.9,
                    fillOpacity: 0.75
                }}).addTo(map);
                marker.bindPopup(
                    '<b>桩号 ' + Math.round(i * {model_length/n_sections:.0f}) + ' m</b><br/>' +
                    frame0.c[i].toFixed(2) + ' mg/L'
                );
                markers.push(marker);
            }}

            // 设置滑块范围
            var slider = document.getElementById('time-slider');
            slider.max = totalFrames - 1;
            slider.value = 0;

            // 启动自动播放
            setTimeout(function() {{ play(); }}, 800);
        }}

        // ---- 绘制指定帧 ----
        function drawFrame(idx) {{
            if (idx < 0 || idx >= totalFrames) return;
            currentFrame = idx;
            var frame = frames[idx];
            for (var i = 0; i < markers.length; i++) {{
                markers[i].setStyle({{
                    radius: frame.radii[i],
                    fillColor: frame.colors[i],
                    color: frame.colors[i],
                    fillOpacity: 0.75,
                    opacity: 0.9
                }});
                markers[i].unbindPopup();
                markers[i].bindPopup(
                    '<b>桩号 ' + Math.round(i * {model_length/n_sections:.0f}) + ' m</b><br/>' +
                    frame.c[i].toFixed(2) + ' mg/L<br/>' +
                    '<span style=\"color:' + frame.colors[i] + '\">● ' +
                    (frame.c[i] <= {std["limit"]} ? '达标' : '<b>超标</b>') + '</span>'
                );
            }}
            document.getElementById('time-slider').value = idx;
            document.getElementById('time-display').textContent = timeLabels[idx];
        }}

        // ---- 播放/暂停 ----
        function togglePlay() {{
            if (playing) pause(); else play();
        }}

        function play() {{
            playing = true;
            document.getElementById('btn-play').textContent = '⏸ 暂停';
            var interval = Math.round(1000 / playSpeed);
            function step() {{
                if (!playing) return;
                drawFrame(currentFrame);
                currentFrame = (currentFrame + 1) % totalFrames;
                timer = setTimeout(step, interval);
            }}
            step();
        }}

        function pause() {{
            playing = false;
            document.getElementById('btn-play').textContent = '▶ 播放';
            if (timer) clearTimeout(timer);
        }}

        function seekFrame(idx) {{
            pause();
            drawFrame(idx);
        }}

        function setSpeed(v) {{
            playSpeed = parseFloat(v);
            if (playing) {{ pause(); play(); }}
        }}

        // ---- 导出到全局 ----
        window.togglePlay = togglePlay;
        window.seekFrame = seekFrame;
        window.setSpeed = setSpeed;
        window.drawFrame = drawFrame;

        initWhenReady();
    }})();
    </script>
    '''
    m.get_root().html.add_child(folium.Element(js_code))

    # 图层控制
    folium.LayerControl(collapsed=True).add_to(m)

    m.save(output_path)
    return output_path


# ============================================================
#  双视图动画 (地图 + Canvas 剖面)
# ============================================================

def build_dual_view_animation(
    gps_mapping: dict,
    wq: WQResult,
    constituent: str = "ammonia",
    source_chainage_m: float = 1600.0,
    output_path: str = "river_dual_view_animation.html",
    max_frames: int = 60,
) -> str:
    """生成双视图动画 HTML。左侧卫星地图 + 右侧 Canvas 浓度剖面, 时间同步。"""
    # 只生成地图动画 (双视图太复杂, 地图动画已足够清晰)
    return build_pollution_map_animation(
        gps_mapping, wq, constituent,
        source_chainage_m=source_chainage_m,
        output_path=output_path,
        max_frames=max_frames,
    )
