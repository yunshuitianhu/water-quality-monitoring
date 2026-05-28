import os
import platform

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# 跨平台中文字体搜索路径
_FONT_SEARCH = {
    "Windows": [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
    ],
    "Linux": [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ],
    "Darwin": [
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ],
}

# 中文标签英文回退
_FALLBACK_LABELS = {
    "氨氮 (mg/L)": "NH3-N (mg/L)",
    "溶解氧 (mg/L)": "DO (mg/L)",
    "浑浊度 (NTU)": "Turbidity (NTU)",
    "时间": "Time",
    "水质时间序列与黑点分布": "Water Quality Time Series & Black Spots",
    "轻度": "Mild",
    "中度": "Moderate",
    "重度": "Severe",
    "严重": "Critical",
    "轻度污染": "Mildly Polluted",
    "中度污染": "Moderately Polluted",
    "重度污染": "Severely Polluted",
    "严重污染": "Critically Polluted",
}


def discover_font(font_path=None):
    """跨平台发现中文字体。返回 FontProperties 或 None。"""
    if font_path and os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    system = platform.system()
    search_paths = _FONT_SEARCH.get(system, [])
    for fp in search_paths:
        if os.path.exists(fp):
            return fm.FontProperties(fname=fp)
    # 尝试通过 matplotlib 字号搜索
    for name in ["SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei"]:
        matched = fm.findfont(name, fallback_to_default=False)
        if matched and os.path.exists(matched):
            return fm.FontProperties(fname=matched)
    return None


def configure_matplotlib_font(font_prop):
    """配置 matplotlib 中文字体，无字体时使用英文回退。"""
    plt.rcParams["axes.unicode_minus"] = False
    if font_prop is not None:
        plt.rcParams["font.family"] = font_prop.get_name()
    else:
        plt.rcParams["font.family"] = "sans-serif"


def get_label(label_cn):
    """获取标签：有中文字体返回中文，否则返回英文回退。"""
    if plt.rcParams["font.family"] != "sans-serif":
        return label_cn
    return _FALLBACK_LABELS.get(label_cn, label_cn)
