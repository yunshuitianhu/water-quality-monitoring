from dataclasses import dataclass, field
import sys
import os

import pandas as pd

# 确保 river_model 在 Python 路径中
_river_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _river_model_path not in sys.path:
    sys.path.insert(0, _river_model_path)


@dataclass
class WaterQualityState:
    df: pd.DataFrame | None = None
    black_spots: pd.DataFrame | None = None
    video_frames: list[dict] = field(default_factory=list)
    cross_validation_result: dict | None = None

    def has_data(self) -> bool:
        return self.df is not None and len(self.df) > 0

    def has_black_spots(self) -> bool:
        return self.black_spots is not None and len(self.black_spots) > 0

    def has_video_frames(self) -> bool:
        return len(self.video_frames) > 0


# 在模块级别导入 RiverModelState (延迟以避免循环)
_RIVER_STATE = None


def _get_river_state():
    """获取或创建 RiverModelState 单例。"""
    global _RIVER_STATE
    if _RIVER_STATE is None:
        from river_model.state import RiverModelState
        _RIVER_STATE = RiverModelState()
    return _RIVER_STATE
