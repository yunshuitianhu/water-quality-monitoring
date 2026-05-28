"""污染溯源动画 — 可调用函数版本。供 Streamlit app.py 和 MCP 导入。

Usage:
    from build_animation import build_trace_animation
    html_path = build_trace_animation()  # 生成 HTML, 返回路径
"""
import json, os, sys
import numpy as np
import pandas as pd

_AIROOT = os.path.dirname(os.path.abspath(__file__))
if _AIROOT not in sys.path:
    sys.path.insert(0, _AIROOT)

from river_model.config import YayaoConfig
from river_model.cross_sections import generate_yayao_cross_sections
from river_model.pure_py_hydro import solve_hydrodynamics
from river_model.wq_engine import solve_water_quality
from river_model.geo_mapping import (build_chainage_gps_mapping, chainage_array_to_gps,
    get_gps_track_coords, get_map_center, chainage_to_gps)
from river_model.animation import _build_color_scale, _conc_to_color_relative, _conc_to_radius_relative
from water_quality_mcp.src.water_quality_mcp.analysis import find_black_spots, calc_black_score_vectorized

EXCEL_DEFAULT = os.path.join(_AIROOT,
    "苗圃杯/无人船智能水质监测分析和因果溯源样本数据/河道巡航走航样本数据.xlsx")

def _load_amap_key():
    cfg_path = os.path.join(_AIROOT, ".mcp.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)["mcpServers"]["water-quality-mcp"]["env"].get("AMAP_KEY", "")
    return ""

def build_trace_animation(
    excel_path: str = None,
    output_path: str = "river_pollution_animation.html",
    constituent: str = "ammonia",
    amap_key: str = None,
    progress_callback = None,  # fn(msg) for Streamlit progress
) -> str:
    """主函数: 生成污染溯源动画 HTML。

    Returns: output_path (str)
    """
    if excel_path is None:
        excel_path = EXCEL_DEFAULT
    if amap_key is None:
        amap_key = _load_amap_key()
    if progress_callback is None:
        progress_callback = print

    progress_callback("1/5 加载数据 + 定位污染源...")

    df = pd.read_excel(excel_path)
    col_map = {"氨氮(mg/L)": "ammonia_nitrogen", "溶解氧(mg/L)": "dissolved_oxygen",
               "经度": "longitude", "纬度": "latitude", "化学需氧量(mg/L)": "cod",
               "pH": "ph", "浑浊度(NTU)": "turbidity"}
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    for c in ["ammonia_nitrogen", "dissolved_oxygen", "cod"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["ammonia_nitrogen"] < 50.0]

    result_json, black_df = find_black_spots(df)
    result = json.loads(result_json)
    clusters = result.get("聚类详情", [])
    worst = max(clusters, key=lambda c: c.get("内梅罗指数_均值", c.get("avg_score", 0))) if clusters else None
    if worst:
        source_lat, source_lon = worst["center_lat"], worst["center_lon"]
    else:
        mx = df["ammonia_nitrogen"].idxmax()
        source_lat, source_lon = float(df.loc[mx, "latitude"]), float(df.loc[mx, "longitude"])
    ammonia_peak = float(result.get("氨氮峰值(mg/L)", 35.8))

    # Search factories
    nearby_factories = []
    if amap_key:
        from water_quality_mcp.src.water_quality_mcp.geo_tools import search_nearby_pollution_sources
        nearby = search_nearby_pollution_sources(source_lat, source_lon, amap_key=amap_key, radius=1000)
        factory_kw = ["工厂","公司","企业","养殖","化工","工业","包装","铝","金属","食品"]
        nearby_factories = [p for p in nearby
            if any(k in str(p.get("type","")) or k in str(p.get("name","")) for k in factory_kw)]
        nearby_factories.sort(key=lambda p: int("".join(filter(str.isdigit, str(p.get("distance","9999")))) or "9999"))
        nearby_factories = nearby_factories[:5]

    if nearby_factories:
        top = nearby_factories[0]
        fd = int("".join(filter(str.isdigit, str(top.get("distance","9999")))) or "9999")
        source_name = top["name"]
        if fd < 500:
            source_lat = top.get("lat", source_lat)
            source_lon = top.get("lon", source_lon)
    else:
        source_name = f"推定污染源(聚类{worst['cluster_id']})" if worst else "氨氮峰值点"

    progress_callback(f"2/5 GPS 映射... | 源: {source_name}")

    mapping = build_chainage_gps_mapping(excel_path, smoothing_sigma=3.0)
    gps_lats, gps_lons = mapping["lat_raw"], mapping["lon_raw"]
    gps_chainage = mapping["chainage_raw"]
    gps_length = mapping["total_length_m"]
    dists = np.sqrt((gps_lats-source_lat)**2 + (gps_lons-source_lon)**2)
    src_chainage = float(gps_chainage[int(np.argmin(dists))])

    est_load = min(ammonia_peak * 6.9 * 1800.0 / 1000.0, 500.0)

    progress_callback("3/5 非恒定流 + ADR...")
    config = YayaoConfig(n_cross_sections=30, mannings_n=0.028, length_m=gps_length)
    cs_list = generate_yayao_cross_sections(30, gps_length, config.bed_elevation_avg_m,
                                              config.longitudinal_slope, 40.0, mannings_n=0.028)
    try:
        hydro = solve_hydrodynamics(config, cs_list, upstream_flow_m3s=6.9,
            downstream_stage_m=0.83, tidal_amplitude_m=0.8, tidal_period_h=12.42,
            duration_h=6.0, dt_s=30.0, output_interval_min=15.0)
        for arr_name in ['velocity','discharge','water_level','area']:
            arr = getattr(hydro, arr_name)
            if arr is not None and np.any(np.isnan(arr)):
                setattr(hydro, arr_name, pd.DataFrame(arr).ffill().bfill().values)
    except Exception:
        from river_model.hydro_engine import run_hydrodynamic
        hydro = run_hydrodynamic(config, cs_list, duration_h=6.0, dt_s=120.0)

    wq = solve_water_quality(config, hydro, pollutant_type="ammonia", load_kg=est_load,
        chainage_m=src_chainage, duration_min=30.0, simulation_hours=24.0, dt_s=15.0)

    progress_callback("4/5 构建帧...")
    model_chainage = wq.chainage
    model_length = float(model_chainage[-1] or gps_length)
    scaled_chainage = model_chainage * (gps_length / model_length)

    gps_grid_ch = np.linspace(0, gps_length, 80)
    gps_grid_lat, gps_grid_lon = chainage_array_to_gps(mapping, gps_grid_ch)

    track_lat = mapping["lat_grid"]; track_lon = mapping["lon_grid"]; track_ch = mapping["chainage_grid"]
    perp_lat = []; perp_lon = []
    for ch in gps_grid_ch:
        idx = int(np.argmin(np.abs(track_ch-ch)))
        dy = track_lat[idx+1]-track_lat[idx] if idx<len(track_ch)-1 else track_lat[idx]-track_lat[idx-1]
        dx = track_lon[idx+1]-track_lon[idx] if idx<len(track_ch)-1 else track_lon[idx]-track_lon[idx-1]
        mag = np.sqrt(dy**2+dx**2)+1e-10
        perp_lat.append(dx/mag); perp_lon.append(-dy/mag)
    perp_lat = np.array(perp_lat); perp_lon = np.array(perp_lon)

    lat2m=111000.0; lon2m=111000.0*np.cos(np.radians(np.mean(gps_grid_lat)))
    hw_lat=27.5/lat2m; hw_lon=27.5/lon2m
    n_grid = len(gps_grid_ch)
    all_lats = np.concatenate([gps_grid_lat,
        gps_grid_lat+perp_lat*hw_lat*0.5, gps_grid_lat-perp_lat*hw_lat*0.5])
    all_lons = np.concatenate([gps_grid_lon,
        gps_grid_lon+perp_lon*hw_lon*0.5, gps_grid_lon-perp_lon*hw_lon*0.5])

    data = wq.ammonia
    nt = len(wq.t)
    frame_indices = np.linspace(0, nt-1, min(nt, 300), dtype=int)
    baseline_conc = data[0,:].copy()
    frames_raw = []
    for fi in frame_indices:
        excess = np.maximum(data[fi,:]-baseline_conc, 0.0)
        cc = np.interp(gps_grid_ch, scaled_chainage, excess)
        cc = np.maximum(cc, 0.0)
        c_all = np.concatenate([cc, cc*0.5, cc*0.5])
        frames_raw.append({"t":round(float(wq.t[fi]),2), "c":[round(float(x),3) for x in c_all]})

    cs = _build_color_scale(frames_raw)
    frames_data = [{"t":f["t"],"c":f["c"],
        "colors":[_conc_to_color_relative(x,cs) for x in f["c"]],
        "radii":[_conc_to_radius_relative(x,cs) for x in f["c"]]} for f in frames_raw]

    time_labels = [f"排放后 {round(float(wq.t[fi]),1):.1f}h" for fi in frame_indices]
    track_coords = get_gps_track_coords(mapping)
    center = get_map_center(mapping)

    # Smart real-data sampling
    valid = df[["latitude","longitude","ammonia_nitrogen"]].dropna()
    valid = valid[valid["ammonia_nitrogen"]<50.0]
    lats_r = valid["latitude"].values.astype(float)
    lons_r = valid["longitude"].values.astype(float)
    concs_r = valid["ammonia_nitrogen"].values.astype(float)
    df_tmp = valid.copy().reset_index(drop=True)
    for cc in ["dissolved_oxygen","cod","turbidity","ph"]:
        if cc in df.columns: df_tmp[cc] = df.loc[valid.index, cc].reset_index(drop=True)
    scores = calc_black_score_vectorized(df_tmp)

    def acolor(c):
        if c<=0.15: return "#0066FF"
        elif c<=0.5: return "#00CC44"
        elif c<=1.0: return "#FFCC00"
        elif c<=1.5: return "#FF8800"
        elif c<=2.0: return "#FF0000"
        else: return "#990000"

    real_data = []
    for i in range(len(lats_r)):
        s = float(scores[i])
        if s >= 0.8:
            if i % 6 == 0:
                real_data.append({"lat":float(lats_r[i]),"lon":float(lons_r[i]),
                    "conc":float(concs_r[i]),"color":acolor(concs_r[i]),"radius":3.5})
        elif i % 20 == 0:
            real_data.append({"lat":float(lats_r[i]),"lon":float(lons_r[i]),
                "conc":float(concs_r[i]),"color":acolor(concs_r[i]),"radius":2.0})

    ammonia_mean = float(np.mean(concs_r))
    over_v_pct = float(np.sum(concs_r>2.0)/len(concs_r)*100)
    peak_sim = max(max(f["c"]) for f in frames_data)

    # Other sources
    other_sources = []
    for fac in nearby_factories[:5]:
        d2 = np.sqrt((gps_lats-fac.get("lat",source_lat))**2+(gps_lons-fac.get("lon",source_lon))**2)
        fch = float(gps_chainage[int(np.argmin(d2))])
        other_sources.append({"name":fac.get("name","?")[:15],"chainage":fch,
            "lat":fac.get("lat",source_lat),"lon":fac.get("lon",source_lon),
            "distance":fac.get("distance","?")})

    factory_list_html = "".join(
        f'<div>• {f.get("name","?")[:15]} <span style="color:#FFCC00">({f.get("distance","?")}m)</span></div>'
        for f in nearby_factories[:5])

    progress_callback(f"5/5 生成 HTML... ({len(frames_data)} 帧)")

    # JSON
    frames_json = json.dumps(frames_data, ensure_ascii=False)
    labels_json = json.dumps(time_labels, ensure_ascii=False)
    lats_json = json.dumps([float(x) for x in all_lats])
    lons_json = json.dumps([float(x) for x in all_lons])
    track_json = json.dumps([[float(x),float(y)] for x,y in track_coords])
    real_json = json.dumps(real_data, ensure_ascii=False)
    factories_json = json.dumps([{"name":f.get("name","?"),"type":f.get("type","?"),
        "address":f.get("address","")[:60],"distance":f.get("distance","?"),
        "lat":f.get("lat",source_lat),"lon":f.get("lon",source_lon)} for f in nearby_factories], ensure_ascii=False)
    other_src_json = json.dumps(other_sources, ensure_ascii=False)
    cs_max = cs['max']

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>雅瑶水道 污染溯源</title><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css"/>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:'Microsoft YaHei',Arial,sans-serif;overflow:hidden}}
#map{{width:100vw;height:100vh}}
#title{{position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:10000;background:rgba(0,0,0,0.85);color:white;padding:8px 24px;border-radius:8px;text-align:center;pointer-events:none}}
#title h3{{margin:0 0 2px;font-size:16px}}#title span{{font-size:12px;opacity:.9}}
#story{{position:fixed;top:90px;right:10px;z-index:10000;background:rgba(0,0,0,0.82);color:#eee;padding:10px 14px;border-radius:8px;font-size:11px;max-width:230px;line-height:1.4}}
#story h4{{margin:0 0 4px;font-size:13px;color:#FFCC00}}
#story .val{{font-weight:bold;color:#FF4500}}#story .ok{{color:#0f0}}
#ctrl{{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);z-index:10000;background:rgba(0,0,0,0.88);color:white;padding:10px 20px;border-radius:12px;display:flex;align-items:center;gap:14px;min-width:520px}}
#ctrl button{{border:none;border-radius:4px;cursor:pointer;font-size:13px;font-weight:bold}}
#btn-play{{background:#e94560;color:white;padding:8px 20px;min-width:76px}}#btn-reset{{background:#555;color:white;padding:8px 12px}}
#time-slider{{flex:1;accent-color:#e94560;height:6px;min-width:100px}}
#time-display{{font-size:15px;font-weight:bold;color:#FFCC00;min-width:100px;text-align:center}}
#speed-select{{background:#333;color:white;border:1px solid #666;border-radius:3px;padding:4px}}#status{{color:#0f0;font-size:11px}}
.leg{{position:fixed;bottom:80px;left:10px;z-index:10000;background:rgba(255,255,255,0.93);padding:8px 10px;border-radius:6px;font-size:11px;box-shadow:0 2px 6px rgba(0,0,0,0.3);max-width:200px}}
.leg-item{{display:flex;align-items:center;margin:1px 0}}.leg-s{{width:14px;height:10px;border-radius:2px;margin-right:4px;flex-shrink:0}}
</style></head><body>
<div id="map"></div>
<div id="title"><h3>雅瑶水道 氨氮污染溯源推演</h3><span>主源: {source_name} | 排放≈{est_load:.0f}kg | 非恒定流+潮汐+热力图 | {len(frames_data)}帧</span></div>
<div id="story"><h4>溯源分析</h4>
<div>主源: <span class="val">{source_name[:20]}</span></div><div>桩号: <span class="val">{src_chainage:.0f}m</span></div>
<hr style="margin:3px 0;border-color:#555"><div style="font-size:10px;color:#FF8800">嫌疑工厂(多源):</div>{factory_list_html}
<hr style="margin:3px 0;border-color:#555"><div style="font-size:10px;color:#FFCC00">⚠上下游均有工厂<br/>实测污染来自多方排放<br/>本动画仅模拟主源</div>
<hr style="margin:4px 0;border-color:#555"><div>氨氮均值:<span class="val">{ammonia_mean:.1f}mg/L</span></div>
<div>超V类:<span class="val">{over_v_pct:.0f}%</span></div><div>模拟峰值:<span class="val">{peak_sim:.1f}mg/L</span></div>
<hr style="margin:4px 0;border-color:#444"><div id="anim-note" style="color:#0f0">准备播放...</div></div>
<div class="leg"><div style="font-weight:bold;margin-bottom:3px">模拟热力图 (增量)</div>
<div class="leg-item"><span class="leg-s" style="background:#0066FF"></span>基准/清洁</div>
<div class="leg-item"><span class="leg-s" style="background:#00CC44"></span>轻微</div>
<div class="leg-item"><span class="leg-s" style="background:#FFCC00"></span>中等</div>
<div class="leg-item"><span class="leg-s" style="background:#FF6600"></span>较重</div>
<div class="leg-item"><span class="leg-s" style="background:#FF0000"></span>严重</div>
<hr style="margin:3px 0"><div style="font-weight:bold;margin-bottom:2px">实测(GB3838)</div>
<div class="leg-item"><span class="leg-s" style="background:#990000"></span>劣V类 &gt;2.0</div><div class="leg-item"><span class="leg-s" style="background:#FF0000"></span>V类 ≤2.0</div></div>
<div id="ctrl"><button id="btn-play" onclick="togglePlay()">播放</button><button id="btn-reset" onclick="seekFrame(0)">重置</button>
<input type="range" id="time-slider" min="0" max="{len(frames_data)-1}" value="0" step="1" oninput="seekFrame(parseInt(this.value))">
<span id="time-display">排放后 0.0h</span><select id="speed-select" onchange="setSpeed(this.value)">
<option value="0.5">0.5x</option><option value="1">1x</option><option value="2" selected>2x</option><option value="5">5x</option><option value="10">10x</option></select><span id="status">加载中...</span></div>
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/leaflet.heat@0.2.0/dist/leaflet-heat.min.js"></script><script>
var frames={frames_json},timeLabels={labels_json},lats={lats_json},lons={lons_json},trackCoords={track_json},center=[{center[0]:.6f},{center[1]:.6f}],srcLat={source_lat:.6f},srcLon={source_lon:.6f},totalFrames=frames.length,modelLengthM={gps_length:.0f},nSections=lats.length,realPoints={real_json},nearbyFactories={factories_json},otherSources={other_src_json},srcChainageM={src_chainage:.0f},estLoad={est_load:.0f},ammoniaPeak={ammonia_peak:.1f};
var cf=0,playing=!1,timer=null,speed=2,realMarkers=[],heatLayer=null,STORAGE_KEY='yayao_anim_v2';
	function saveState(){{try{{localStorage.setItem(STORAGE_KEY,JSON.stringify({{f:cf,p:playing,s:speed}}))}}catch(e){{}}}}
	function loadState(){{try{{var s=JSON.parse(localStorage.getItem(STORAGE_KEY));if(s&&typeof s.f==='number'){{cf=Math.min(s.f,totalFrames-1);speed=s.s||2;document.getElementById('speed-select').value=speed;document.getElementById('status').textContent='已恢复上次位置(帧'+(cf+1)+'/'+totalFrames+')';return true}}}}catch(e){{}}return false}}
var map=L.map('map',{{center:center,zoom:15}});
var TILE_SOURCES=['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}','https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}','https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png','https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}.png'];
var tileLayer=L.tileLayer(TILE_SOURCES[0],{{attribution:'Esri|OSM|CartoDB',maxZoom:19,subdomains:'abc'}});
tileLayer.on('tileerror',function(e){{var t=e.tile,c=e.coords;function trySrc(i){{if(i>=TILE_SOURCES.length)return;var u=TILE_SOURCES[i].replace('{{z}}',c.z).replace('{{x}}',c.x).replace('{{y}}',c.y).replace('{{s}}','abc'[i%3]);var x=new Image();x.onload=function(){{t.src=u}};x.onerror=function(){{trySrc(i+1)}};x.src=u}}trySrc(1)}});
tileLayer.addTo(map);
L.polyline(trackCoords,{{color:'#0066CC',weight:5,opacity:.7}}).addTo(map);
L.marker(trackCoords[0]).addTo(map).bindPopup('上游:北村水闸');L.marker(trackCoords[trackCoords.length-1]).addTo(map).bindPopup('下游:珠江');
for(var i=0;i<realPoints.length;i++){{var p=realPoints[i],m=L.circleMarker([p.lat,p.lon],{{radius:p.radius,fillColor:p.color,color:p.color,weight:0,opacity:.25,fillOpacity:.2}}).addTo(map);m.bindPopup('实测:'+p.conc.toFixed(1)+' mg/L');realMarkers.push(m)}}
var srcIcon=L.divIcon({{html:'<div style="background:#F00;color:white;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:14px;border:3px solid #fff;box-shadow:0 0 10px red">!</div>',className:'',iconSize:[28,28],iconAnchor:[14,14]}});
L.marker([srcLat,srcLon],{{icon:srcIcon}}).addTo(map).bindPopup('<b>主源:'+nearbyFactories[0].name+'</b><br/>桩号:'+srcChainageM+'m<br/>估算排放:'+estLoad.toFixed(0)+'kg');
var pulseRing=L.circleMarker([srcLat,srcLon],{{radius:20,color:'#F00',fillOpacity:0,weight:3,opacity:.6}}).addTo(map);
for(var i=0;i<otherSources.length;i++){{var os=otherSources[i],oIcon=L.divIcon({{html:'<div style="background:#FF8800;color:white;border-radius:50%;width:18px;height:18px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:9px;border:2px solid #fff">'+(i+2)+'</div>',className:'',iconSize:[18,18],iconAnchor:[9,9]}});L.marker([os.lat,os.lon],{{icon:oIcon}}).addTo(map).bindPopup('<b>'+os.name+'</b><br/>桩号:'+Math.round(os.chainage)+'m<br/>距离聚类:'+os.distance+'m')}}
function buildHeatData(idx){{if(idx<0||idx>=totalFrames)return[];var f=frames[idx],points=[];for(var i=0;i<nSections;i++){{var intensity=Math.min(f.c[i]/{cs_max:.1f},1.0);if(intensity>0.001)points.push([lats[i],lons[i],intensity])}}return points}}
var initData=buildHeatData(0);heatLayer=L.heatLayer(initData,{{radius:25,blur:15,maxZoom:17,max:1.0,gradient:{{0:'#0066FF',0.25:'#00CC44',0.5:'#FFCC00',0.75:'#FF6600',1:'#FF0000'}},minOpacity:0.3}}).addTo(map);
document.getElementById('status').textContent='就绪(热力图,'+realPoints.length+'实测点)';
function drawFrame(idx){{if(idx<0||idx>=totalFrames)return;cf=idx;var f=frames[idx];heatLayer.setLatLngs(buildHeatData(idx));var maxC=Math.max.apply(null,f.c);document.getElementById('time-slider').value=idx;document.getElementById('time-display').textContent=timeLabels[idx];document.getElementById('anim-note').innerHTML='时间:<span class="val">'+f.t.toFixed(1)+'h</span><br/>峰值:<span class="val">'+maxC.toFixed(1)+'mg/L</span>';saveState()}}
function togglePlay(){{playing?pause():play()}}function play(){{playing=!0;document.getElementById('btn-play').textContent='暂停';document.getElementById('btn-play').style.background='#00aa44';document.getElementById('anim-note').style.color='#FF4500';function step(){{if(!playing)return;drawFrame(cf);cf=(cf+1)%totalFrames;timer=setTimeout(step,Math.round(1000/speed))}}step()}}
function pause(){{playing=!1;document.getElementById('btn-play').textContent='播放';document.getElementById('btn-play').style.background='#e94560';document.getElementById('anim-note').style.color='#0f0';if(timer)clearTimeout(timer);saveState()}}
function seekFrame(idx){{pause();drawFrame(idx);saveState()}}function setSpeed(v){{speed=parseFloat(v);saveState();if(playing){{pause();play()}}}}
setInterval(function(){{var r=pulseRing.getRadius();pulseRing.setRadius(r>22?16:r+.5);pulseRing.setStyle({{opacity:.3+(r-16)/12}})}},100);
if(!loadState()){{setTimeout(function(){{play()}},2000)}}else{{drawFrame(cf)}}
document.addEventListener('keydown',function(e){{if(e.key===' '){{e.preventDefault();togglePlay()}}if(e.key==='ArrowLeft')seekFrame(Math.max(0,cf-1));if(e.key==='ArrowRight')seekFrame(Math.min(totalFrames-1,cf+1))}});
drawFrame(0);
</script></body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    progress_callback(f"完成: {output_path} ({os.path.getsize(output_path)/1024:.0f} KB)")
    return output_path


# ====== CLI entry point ======
if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    out = build_trace_animation()
    print(f"Done: {out}")
    os.startfile(out) if os.name == "nt" else None
