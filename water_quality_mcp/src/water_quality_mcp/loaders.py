import base64
import io
import os
import tempfile

import cv2
import numpy as np
import pandas as pd


def load_data(file_bytes: str, filename: str):
    """加载并清洗 CSV/Excel 走航数据。返回 (df, report_string)。"""
    raw = base64.b64decode(file_bytes)
    buf = io.BytesIO(raw)
    fn = filename.lower()
    try:
        if fn.endswith('.csv'):
            try:
                df = pd.read_csv(buf, encoding='utf-8')
            except UnicodeDecodeError:
                buf.seek(0)
                df = pd.read_csv(buf, encoding='gbk')
        elif fn.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(buf)
        else:
            return None, "不支持的文件格式，请上传 .csv / .xls / .xlsx"
    except Exception as e:
        return None, f"读取失败: {e}"

    col_map = {
        '时间': 'timestamp', '氨氮(mg/L)': 'ammonia_nitrogen', '溶解氧(mg/L)': 'dissolved_oxygen',
        '水温(°C)': 'water_temperature', '经度': 'longitude', '纬度': 'latitude', '电导率': 'conductivity',
        '化学需氧量(mg/L)': 'cod', '总有机碳(mg/L)': 'toc', 'pH': 'ph', '浑浊度(NTU)': 'turbidity'
    }
    df.rename(columns=col_map, inplace=True)
    ts = df['timestamp'].astype(str).str.strip().str.replace('　', ' ').str.replace('：', ':')
    dt = pd.to_datetime(ts, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    mask = dt.isna()
    if mask.any():
        short = ts[mask].str.slice(0, 16)
        dt[mask] = pd.to_datetime(short, format='%Y-%m-%d %H:%M', errors='coerce')
    df['datetime'] = dt
    df.dropna(subset=['datetime'], inplace=True)
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    numeric_cols = ['ammonia_nitrogen', 'dissolved_oxygen', 'water_temperature', 'cod', 'turbidity', 'ph']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['ph', 'dissolved_oxygen', 'water_temperature']:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    limits = {'ammonia_nitrogen': (0, 1000), 'dissolved_oxygen': (0, 20), 'water_temperature': (0, 45),
              'cod': (0, 500), 'turbidity': (0, 2000), 'ph': (3, 12)}
    for col, (lo, hi) in limits.items():
        if col in df.columns:
            df[col] = df[col].where((df[col] >= lo) & (df[col] <= hi), np.nan)

    for col in numeric_cols:
        if col in df.columns and df[col].isna().sum() > 0:
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
    for col in ['longitude', 'latitude']:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    report = [f"数据加载完成，共 {len(df)} 条记录"]
    for col in ['ammonia_nitrogen', 'dissolved_oxygen', 'turbidity', 'ph']:
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 0:
                report.append(f"   {col}: 有效值={len(valid)}, 均值={valid.mean():.2f}, 最小={valid.min():.2f}, 最大={valid.max():.2f}")
    return df, "\n".join(report)


def extract_video(file_bytes: str, interval: int = 10):
    """从视频文件中抽帧。返回 (frames_list, report_string)。"""
    raw = base64.b64decode(file_bytes)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tmp.write(raw)
    vpath = tmp.name
    tmp.close()

    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
    cap = cv2.VideoCapture(vpath, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        os.unlink(vpath)
        return [], "无法打开视频文件，请检查格式"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frame = max(1, int(fps * interval))
    count, frames, skipped = 0, [], 0
    while True:
        try:
            ret, frame = cap.read()
        except Exception:
            skipped += 1
            count += 1
            continue
        if not ret:
            break
        if count % interval_frame == 0:
            if frame is not None and frame.size > 0 and frame.mean() > 1.0:
                _, buf = cv2.imencode('.jpg', frame)
                frames.append({
                    'time_sec': round(count / fps, 1),
                    'image_base64': base64.b64encode(buf).decode()
                })
            else:
                skipped += 1
        count += 1
    cap.release()
    os.unlink(vpath)
    return frames, f"视频抽帧完成，共 {len(frames)} 帧（总 {total_frames} 帧，跳过 {skipped} 个损坏帧）"
