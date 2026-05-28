import json

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from folium.plugins import HeatMap
from matplotlib.patches import Patch

from .analysis import get_severity_info, get_cluster_extent
from .font_utils import configure_matplotlib_font, discover_font, get_label


STARCLOUD_TOKEN_DEFAULT = None  # 由环境变量 CLOUD_TOKEN 提供


def generate_time_series(state, output_path="time_series.png", font_path=None):
    """生成 4 面板时间序列图（氨氮/溶解氧/浊度/pH），黑点按严重等级着色。"""
    df = state.df
    black = state.black_spots
    if df is None or len(df) == 0:
        return "数据未加载"

    font_prop = discover_font(font_path)
    configure_matplotlib_font(font_prop)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # 氨氮
    axes[0].plot(df['datetime'], df['ammonia_nitrogen'], linewidth=0.8)
    axes[0].axhline(2.0, color='r', ls='--')
    if black is not None and len(black) > 0:
        sc_colors = black['severity_color'].values if 'severity_color' in black.columns else 'red'
        axes[0].scatter(black['datetime'], black['ammonia_nitrogen'], c=sc_colors, s=10, edgecolors='black', linewidths=0.2)
    axes[0].set_ylabel(get_label('氨氮 (mg/L)'), fontproperties=font_prop)
    axes[0].grid(True, alpha=0.3)

    # 溶解氧
    axes[1].plot(df['datetime'], df['dissolved_oxygen'], linewidth=0.8)
    axes[1].axhline(2.0, color='r', ls='--')
    if black is not None and len(black) > 0:
        sc_colors = black['severity_color'].values if 'severity_color' in black.columns else 'red'
        axes[1].scatter(black['datetime'], black['dissolved_oxygen'], c=sc_colors, s=10, edgecolors='black', linewidths=0.2)
    axes[1].set_ylabel(get_label('溶解氧 (mg/L)'), fontproperties=font_prop)
    axes[1].grid(True, alpha=0.3)

    # 浊度
    if 'turbidity' in df.columns:
        axes[2].plot(df['datetime'], df['turbidity'], linewidth=0.8)
        if black is not None and len(black) > 0:
            sc_colors = black['severity_color'].values if 'severity_color' in black.columns else 'red'
            axes[2].scatter(black['datetime'], black['turbidity'], c=sc_colors, s=10, edgecolors='black', linewidths=0.2)
    axes[2].set_ylabel(get_label('浑浊度 (NTU)'), fontproperties=font_prop)
    axes[2].grid(True, alpha=0.3)

    # pH
    if 'ph' in df.columns:
        axes[3].plot(df['datetime'], df['ph'], linewidth=0.8, color='green')
        axes[3].axhline(6.0, color='gray', ls='--')
        axes[3].axhline(9.0, color='gray', ls='--')
        if black is not None and len(black) > 0:
            sc_colors = black['severity_color'].values if 'severity_color' in black.columns else 'red'
            axes[3].scatter(black['datetime'], black['ph'], c=sc_colors, s=10, edgecolors='black', linewidths=0.2)
    axes[3].set_ylabel('pH', fontproperties=font_prop)
    axes[3].set_xlabel(get_label('时间'), fontproperties=font_prop)
    axes[3].grid(True, alpha=0.3)

    legend_patches = [
        Patch(color='#2E8B57', label=get_label('无黑臭 (达V类)')),
        Patch(color='#FFA500', label=get_label('轻度黑臭')),
        Patch(color='#8B0000', label=get_label('重度黑臭')),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=8, framealpha=0.8)
    plt.suptitle(get_label('水质时间序列与黑点分布'), fontsize=14, fontproperties=font_prop)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return f"时间序列图已保存至 {output_path}"


def generate_trace_map(state, output_path="trace_map.html", amap_key=None, cloud_token=None):
    """生成 folium 交互式溯源地图。"""
    df = state.df
    black = state.black_spots
    if df is None or black is None:
        return "数据或黑点未就绪"

    coords = df[['latitude', 'longitude']].dropna().values
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=15, tiles=None)

    # —— 底图（多源 LayerControl 切换，应对部分服务器在国内不可达）——
    # 所有 GPS 走航数据为 WGS-84，默认显示 CartoDB Voyager（CDN 全球加速，国内可达）。
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
    token = cloud_token or STARCLOUD_TOKEN_DEFAULT
    if token and "你的星图" not in str(token):
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

    # 轨迹
    folium.PolyLine(coords, weight=2, color='blue', opacity=0.5).add_to(m)
    folium.map.Marker(coords[0], icon=folium.DivIcon(html='<div style="color:green;font-weight:bold;">起点</div>')).add_to(m)
    folium.map.Marker(coords[-1], icon=folium.DivIcon(html='<div style="color:red;font-weight:bold;">终点</div>')).add_to(m)

    # 热力图
    HeatMap(black[['latitude', 'longitude']].values, radius=20, blur=10).add_to(m)

    # 采样点
    sample = df.iloc[::30]
    for _, row in sample.iterrows():
        nh3 = f"{row['ammonia_nitrogen']:.1f}" if pd.notna(row.get('ammonia_nitrogen')) else '--'
        do = f"{row['dissolved_oxygen']:.1f}" if pd.notna(row.get('dissolved_oxygen')) else '--'
        cod = f"{row['cod']:.0f}" if pd.notna(row.get('cod')) else '--'
        turb = f"{row['turbidity']:.0f}" if pd.notna(row.get('turbidity')) else '--'
        popup_html = f"<b>{row['datetime']}</b><br>氨氮:{nh3} mg/L<br>DO:{do} mg/L<br>COD:{cod} mg/L<br>浊度:{turb} NTU"
        folium.CircleMarker(
            [row['latitude'], row['longitude']], radius=3, color='darkblue', fill=True, fill_opacity=0.6,
            popup=folium.Popup(popup_html, max_width=220)
        ).add_to(m)

    # 聚类
    for cid in sorted(black['cluster_id'].unique()):
        if cid == -1:
            continue
        cluster = black[black['cluster_id'] == cid]
        avg_score = float(cluster['black_score'].mean())
        _, sev_label, sev_color = get_severity_info(avg_score)

        clat, clon, extent_r = get_cluster_extent(cluster)
        folium.Circle([clat, clon], radius=extent_r, color=sev_color, weight=1.5, fill=True,
                      fill_opacity=0.06, dash_array='5,5').add_to(m)

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

        if amap_key:
            from .geo_tools import search_nearby_pollution_sources
            pois = search_nearby_pollution_sources(clat, clon, amap_key)
            for poi in pois[:5]:
                folium.Marker(
                    [poi['lat'], poi['lon']],
                    popup=f"{poi['name']}({poi['type']})<br>{poi.get('address', '')}",
                    icon=folium.Icon(color='black', icon='industry')
                ).add_to(m)

    # 污染最严重 Top-5 个体点位标记
    from .analysis import get_severity_info as _get_sev
    top5 = black.nlargest(5, 'black_score')
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        sev_color = row['severity_color']
        _, sev_label, _ = _get_sev(row['black_score'])
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

    # 图例
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
    m.save(output_path)
    return f"交互式地图已保存至 {output_path}"


# ============================================================
#  视频与监测数据交叉印证图表
# ============================================================

def generate_cross_validation_chart(state, output_path="cross_validation.png",
                                    time_offset_sec=0, font_path=None):
    """双面板时间序列对比图: 视频指标 vs 传感器指标。

    上: Laplacian方差 (绿) vs 传感器浊度 (蓝)
    下: HSV亮度 (绿) vs 传感器氨氮 (红)
    灰色底色 = 视频判黑区域, 红色底色 = 传感器判黑区域
    """
    from .analysis import _analyze_video_frames_structured, _determine_black_odor_vectorized

    df = state.df
    frames = state.video_frames if hasattr(state, 'video_frames') else []
    if df is None or len(df) == 0 or not frames:
        return "数据或视频帧未就绪"

    font_prop = discover_font(font_path)
    configure_matplotlib_font(font_prop)

    vid_df = _analyze_video_frames_structured(frames)
    if vid_df.empty:
        return "未能从视频帧中提取有效指标"

    from datetime import timedelta
    base_time = df['datetime'].min() + timedelta(seconds=time_offset_sec)
    vid_df['frame_datetime'] = vid_df['time_sec'].apply(lambda t: base_time + timedelta(seconds=t))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # ---- 面板1: 浊度对比 ----
    # 传感器浊度时间序列 (完整)
    if 'turbidity' in df.columns:
        ax1.plot(df['datetime'], df['turbidity'], linewidth=0.6, color='#3388FF', alpha=0.7, label=get_label('传感器浊度 (NTU)'))
    ax1_twin = ax1.twinx()
    ax1_twin.plot(vid_df['frame_datetime'], vid_df['laplacian_var'], linewidth=1.5, color='#00AA44',
                  marker='o', markersize=3, label=get_label('视频 Laplacian 方差'))
    ax1.set_ylabel(get_label('浊度 (NTU)'), fontproperties=font_prop, color='#3388FF')
    ax1_twin.set_ylabel(get_label('Laplacian 方差'), fontproperties=font_prop, color='#00AA44')
    ax1.set_title(get_label('视频视觉清晰度 vs 传感器浊度 (预期负相关)'), fontproperties=font_prop)
    ax1.grid(True, alpha=0.3)

    # 合并图例
    lines1 = ax1.get_lines() + ax1_twin.get_lines()
    labels1 = [l.get_label() for l in lines1]
    ax1.legend(lines1, labels1, loc='upper right', fontsize=8)

    # ---- 面板2: 氨氮对比 ----
    if 'ammonia_nitrogen' in df.columns:
        ax2.plot(df['datetime'], df['ammonia_nitrogen'], linewidth=0.6, color='#E94560', alpha=0.7, label=get_label('传感器氨氮 (mg/L)'))
    # 黑臭阈值线
    ax2.axhline(8.0, color='#E94560', ls='--', alpha=0.5, label='氨氮黑臭阈值 8.0 mg/L')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(vid_df['frame_datetime'], vid_df['mean_brightness'], linewidth=1.5, color='#00AA44',
                  marker='o', markersize=3, label=get_label('视频 HSV 亮度'))
    ax2.set_ylabel(get_label('氨氮 (mg/L)'), fontproperties=font_prop, color='#E94560')
    ax2_twin.set_ylabel(get_label('HSV 亮度 (V)'), fontproperties=font_prop, color='#00AA44')
    ax2.set_title(get_label('视频水色亮度 vs 传感器氨氮 (预期负相关)'), fontproperties=font_prop)
    ax2.set_xlabel(get_label('时间'), fontproperties=font_prop)
    ax2.grid(True, alpha=0.3)

    lines2 = ax2.get_lines() + ax2_twin.get_lines()
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labels2, loc='upper right', fontsize=8)

    # ---- 高亮判黑区域 ----
    # 视频判黑 (color_label == "疑似黑臭")
    black_vid = vid_df[vid_df['color_label'] == '疑似黑臭']
    for _, row in black_vid.iterrows():
        t = row['frame_datetime']
        ax1.axvspan(t - timedelta(seconds=5), t + timedelta(seconds=5), alpha=0.08, color='gray', zorder=0)
        ax2.axvspan(t - timedelta(seconds=5), t + timedelta(seconds=5), alpha=0.08, color='gray', zorder=0)

    # 传感器判黑 (is_black)
    if state.black_spots is not None and len(state.black_spots) > 0:
        black_times = state.black_spots['datetime']
        for t in black_times:
            ax1.axvspan(t - timedelta(seconds=5), t + timedelta(seconds=5), alpha=0.06, color='red', zorder=0)
            ax2.axvspan(t - timedelta(seconds=5), t + timedelta(seconds=5), alpha=0.06, color='red', zorder=0)

    # 图例说明
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color='gray', alpha=0.3, label=get_label('视频判黑区域')),
        Patch(color='red', alpha=0.2, label=get_label('传感器判黑区域')),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=2, fontsize=8, framealpha=0.8)

    plt.suptitle(get_label('视频-传感器交叉印证'), fontsize=14, fontproperties=font_prop)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return f"交叉印证图已保存至 {output_path}"


def generate_correlation_heatmap(corr_json_str, output_path="correlation_heatmap.png", font_path=None):
    """相关系数热力图。输入为 cross_validate_video_with_monitoring 的 JSON 结果。"""
    data = json.loads(corr_json_str) if isinstance(corr_json_str, str) else corr_json_str
    corr_list = data.get("相关系数矩阵", [])
    if not corr_list:
        return "无相关系数数据"

    font_prop = discover_font(font_path)
    configure_matplotlib_font(font_prop)

    v_labels = sorted(set(c['视频指标'] for c in corr_list))
    s_labels = sorted(set(c['传感器指标'] for c in corr_list))
    matrix = np.zeros((len(v_labels), len(s_labels)))
    annot = [["" for _ in s_labels] for _ in v_labels]

    for c in corr_list:
        vi = v_labels.index(c['视频指标'])
        si = s_labels.index(c['传感器指标'])
        matrix[vi, si] = c['Pearson_r']
        sig = c['显著性'].replace('*', '★')
        annot[vi][si] = f"{c['Pearson_r']:.3f}\n{sig}"

    fig, ax = plt.subplots(figsize=(max(6, len(s_labels) * 1.5), max(4, len(v_labels) * 1.2)))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(s_labels)))
    ax.set_xticklabels([get_label(l) for l in s_labels], fontproperties=font_prop, fontsize=10)
    ax.set_yticks(range(len(v_labels)))
    ax.set_yticklabels([get_label(l) for l in v_labels], fontproperties=font_prop, fontsize=10)

    for i in range(len(v_labels)):
        for j in range(len(s_labels)):
            ax.text(j, i, annot[i][j], ha='center', va='center', fontsize=9,
                    color='white' if abs(matrix[i, j]) > 0.5 else 'black')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')
    ax.set_title(get_label('视频指标 × 传感器指标 Pearson 相关系数'), fontproperties=font_prop, fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return f"相关系数热力图已保存至 {output_path}"
