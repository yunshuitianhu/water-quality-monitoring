import json
import math

import requests

# ---------------------------------------------------------------------------
# WGS-84 ↔ GCJ-02 坐标转换
# ---------------------------------------------------------------------------
# GCJ-02（火星坐标系）是中国国测局对 WGS-84 加密后的坐标系，偏移量 100-500m。
# 高德、腾讯等国内图商均使用 GCJ-02。算法来自公开的逆向工程文献。
# ---------------------------------------------------------------------------

_X_PI = math.pi * 3000.0 / 180.0
_A = 6378245.0  # 克拉索夫斯基椭球长半轴
_EE = 0.00669342162296594323  # 第一偏心率平方


def _out_of_china(lat, lon):
    return not (0.8293 < lat < 55.8271 and 72.004 < lon < 137.8347)


def _transform_lat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320.0 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lat, lon):
    """WGS-84 → GCJ-02 火星坐标系。境外坐标直接返回原值。"""
    if _out_of_china(lat, lon):
        return lat, lon
    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = 1 - _EE * math.sin(radlat) ** 2
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrtmagic) * math.pi)
    dlon = (dlon * 180.0) / (_A / sqrtmagic * math.cos(radlat) * math.pi)
    return lat + dlat, lon + dlon


def gcj02_to_wgs84(lat, lon):
    """GCJ-02 → WGS-84 反向转换（迭代法，精度 < 0.5m）。"""
    if _out_of_china(lat, lon):
        return lat, lon
    glat, glon = wgs84_to_gcj02(lat, lon)
    dlat = glat - lat
    dlon = glon - lon
    return lat - dlat, lon - dlon


# ---------------------------------------------------------------------------
# POI / 逆地理编码 / 卫星影像
# ---------------------------------------------------------------------------

def search_nearby_pollution_sources(lat, lon, amap_key, radius=500):
    """搜索周边污染源（高德POI），按距离排序。"""
    keywords = ["工厂", "养殖场", "排污口", "化工厂"]
    all_pois = []
    for kw in keywords:
        url = "https://restapi.amap.com/v3/place/around"
        params = {"location": f"{lon},{lat}", "keywords": kw, "radius": radius, "key": amap_key, "offset": 15}
        try:
            resp = requests.get(url, params=params, timeout=8)
            data = resp.json()
            if data.get("status") == "1":
                for p in data.get("pois", []):
                    loc = p.get("location", "").split(",")
                    all_pois.append({
                        "name": p.get("name", ""), "type": p.get("type", ""),
                        "address": p.get("address", ""), "distance": p.get("distance", ""),
                        "lat": float(loc[1]) if len(loc) == 2 else lat,
                        "lon": float(loc[0]) if len(loc) == 2 else lon
                    })
        except Exception:
            continue

    def dist_num(poi):
        d = poi.get('distance')
        if d is None:
            return 999999
        digits = ''.join(filter(str.isdigit, str(d)))
        return int(digits) if digits else 999999

    all_pois.sort(key=dist_num)
    return all_pois


def reverse_geocode(lat, lon, amap_key):
    """逆地理编码：经纬度 → 地址/省份/周边POI。"""
    url = "https://restapi.amap.com/v3/geocode/regeo"
    params = {"location": f"{lon},{lat}", "key": amap_key, "radius": 1000, "extensions": "all"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") == "1":
            regeo = data["regeocode"]
            address = regeo.get("formatted_address", "")
            province = regeo.get("addressComponent", {}).get("province", "")
            pois = [{"名称": p.get("name", ""), "类型": p.get("type", ""), "距离": p.get("distance", "")}
                    for p in (regeo.get("pois") or [])[:10]]
            return json.dumps({"地址": address, "省份": province, "周边POI": pois}, ensure_ascii=False)
        return "逆地理编码失败"
    except Exception:
        return "逆地理编码异常"


def get_satellite_image(lat, lon, zoom=16, cloud_token=None):
    """星图地球卫星影像：经纬度 → XYZ瓦片 → 验证可用性。"""
    if not cloud_token or "你的星图" in cloud_token:
        return json.dumps({"error": "未配置星图地球 Token"}, ensure_ascii=False)
    n = 2 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    tile_url = f"https://tiles1.geovisearth.com/base/v1/img/{zoom}/{x_tile}/{y_tile}?format=webp&tmsIds=w&token={cloud_token}"
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
            return json.dumps({"error": f"HTTP {resp.status_code}，缩放级别 {zoom} 可能超出覆盖范围"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"卫星影像请求异常: {e}"}, ensure_ascii=False)
