# -*- coding: utf-8 -*-
import csv
import contextlib
import io
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from stiv_adapt.core import _save_img, build_sti_from_frames, init_debug_dir, push_debug_dir, spatial_sample_count

try:
    import pandas as _pandas_probe  # noqa: F401
except ModuleNotFoundError:
    import sys

    class _MissingPandasModule:
        def DataFrame(self, *args: Any, **kwargs: Any) -> Any:
            raise ModuleNotFoundError("No module named 'pandas'")

    sys.modules["pandas"] = _MissingPandasModule()

from stiv_adapt.search import _adaptive_direction_search_on_frames, _load_video_frames
from stiv_adapt.sobel import build_J1_grad_mag, build_circular_roi_mask

t0 = time.perf_counter()

# ========== 用户配置 ==========
VIDEO = r"D:\Desktop\东风渠\测流视频\20260605\170000_undistort.mp4"
CONFIG_PATH = "Config.ini"

# 中文控制台通常使用 GBK。若终端编码为 UTF-8，可改成 "utf-8"。
OUTPUT_ENCODING: Optional[str] = "utf-8"

# 东风渠测速线布设模式：取值 "hydroview" 表示对齐源码布设逻辑；
# 取值 "manual_range" 表示保留之前固定可见起点距范围的逻辑。
TESTING_LINE_LAYOUT_MODE = "hydroview"
DONGFENG_DIST_START_M = 3.1
DONGFENG_DIST_END_M = 20.6
DONGFENG_DIST_STEP_M = 0.5
TEMP_DIR_NAME = "Temp"
TERRAIN_PROFILE_PATH: Optional[str] = r"D:\Desktop\东风渠\260525视频测流软件源码V4.2.6修正虚拟水尺闪退Bug\测流断面地形.txt"

# 时空图像测线参数。LENGTH_PX 仍表示源图像里的测速线长度。
LENGTH_PX = 1500
ANGLE_START, ANGLE_END, ANGLE_STEP = 0, 0, 1
MAX_FRAMES = 750
SPATIAL_SAMPLE_STEP = 2
ARROW_HEAD_PX = 10

START_FRAME: Optional[int] = None
START_TIME_SEC: Optional[float] = None

SCORE_MODE: str = "peak_ratio"
EDGE_METHOD: str = "sobel"
K_SIGMA: float = 2

USE_ROI = True
ROI_RADIUS_FRAC: float = 0.9
ROI: Optional[Tuple[int, int, int, int]] = None
VERBOSE = True
SUPPRESS_LIBRARY_OUTPUT = False

USE_E_ASYM: bool = False
USE_M_MONO: bool = False
M_MONO_PEAK_RATIO: float = 0.5
TOP_K_CANDIDATES: int = 3

DIRECTION_METHOD: str = "hough"
SAVE_ALL_STI: bool = True
SAVE_DEBUG_IMAGES: bool = True

V_MIN: Optional[float] = None
V_MAX: Optional[float] = None
FPS: Optional[float] = None

VOTE_THETA_RES_DEG = 0.1
VOTE_K_RATIO: float = 0.55
VOTE_THETA_RANGE = (70, 87)


# 霍夫法线角 θ 的粗搜到精搜
USE_COARSE_TO_FINE_THETA: bool = True
VOTE_THETA_COARSE_RES_DEG: float = 0.5  #粗搜步长
VOTE_THETA_FINE_RES_DEG: float = 0.05   #精搜步长
VOTE_THETA_FINE_HALF_WIDTH_DEG: float = 1.0   #精搜半宽
# =================================


def configure_output_encoding() -> None:
    """让中文进度日志匹配当前控制台编码。"""
    if not OUTPUT_ENCODING:
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding=OUTPUT_ENCODING, errors="replace")
        except Exception:
            pass


def load_config_ini(path: str) -> Dict[str, Dict[str, str]]:
    """读取 Hydroview 使用的简单 key=value 配置文件。"""
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "gbk", "cp936"):
        try:
            with open(path, "r", encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise RuntimeError(f"failed to decode Config.ini: {last_error}")

    data: Dict[str, Dict[str, str]] = {}
    section = ""
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            data.setdefault(section, {})
            continue
        if "=" not in line or not section:
            continue
        key, value = line.split("=", 1)
        data.setdefault(section, {})[key.strip()] = value.strip()
    return data


def cfg_float(cfg: Dict[str, Dict[str, str]], section: str, key: str, default: Optional[float] = None) -> float:
    try:
        return float(cfg[section][key])
    except KeyError:
        if default is None:
            raise
        return float(default)


def cfg_int(cfg: Dict[str, Dict[str, str]], section: str, key: str, default: Optional[int] = None) -> int:
    try:
        return int(float(cfg[section][key]))
    except KeyError:
        if default is None:
            raise
        return int(default)


class DongfengCameraGeometry:
    """东风渠测速线布设所需的简化水面几何模型。"""

    def __init__(self, cfg: Dict[str, Dict[str, str]], frame_w: int, frame_h: int):
        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)
        self.fx = cfg_float(cfg, "MeasuringSettings", "fx")
        self.fy = cfg_float(cfg, "MeasuringSettings", "fy")
        self.cx = cfg_float(cfg, "MeasuringSettings", "cx")
        self.cy = cfg_float(cfg, "MeasuringSettings", "cy")
        self.img_w = cfg_int(cfg, "MeasuringSettings", "ImgWidth", frame_w)
        self.img_h = cfg_int(cfg, "MeasuringSettings", "ImgHeight", frame_h)
        self.position_to_ref = bool(cfg_int(cfg, "MeasuringSettings", "CamPositionToRef"))

        self.stage_m = cfg_float(cfg, "UserSettings", "StageInput")
        camera_height_m = cfg_float(cfg, "MeasuringSettings", "CameraHeight")
        self.height_above_water_m = camera_height_m - self.stage_m
        if self.height_above_water_m <= 0:
            raise ValueError(
                f"invalid camera height above water: CameraHeight={camera_height_m}, StageInput={self.stage_m}"
            )

        self.camera_dist_m = cfg_float(cfg, "MeasuringSettings", "CameraDist")
        pitch_deg = cfg_float(cfg, "MeasuringSettings", "TiltSensorPitch")
        pitch_deg += cfg_float(cfg, "MeasuringSettings", "dPitch", 0.0)
        roll_deg = cfg_float(cfg, "MeasuringSettings", "TiltSensorRoll", 0.0)
        roll_deg += cfg_float(cfg, "MeasuringSettings", "dRoll", 0.0)
        self.pitch_deg = pitch_deg
        self.roll_deg = roll_deg
        self.pitch_rad = math.radians(pitch_deg)
        self.roll_rad = math.radians(roll_deg)
        self.sensor_size_mm_per_px = cfg_float(cfg, "MeasuringSettings", "s")
        self.focal_length_mm = (self.fx + self.fy) * self.sensor_size_mm_per_px / 2.0

    def station_to_pixel_y(self, station_m: float) -> float:
        """把断面起点距换算为测速线中心 x 处的图像 y 坐标。"""
        forward_m = float(station_m) - self.camera_dist_m
        if forward_m <= 0:
            raise ValueError(f"station {station_m:.3f} m is behind the camera reference")
        beta = math.atan(self.height_above_water_m / forward_m) - self.pitch_rad
        return self.cy + self.fy * math.tan(beta)

    def ground_point(self, u: float, v: float) -> Tuple[float, float]:
        """返回水面物点坐标，格式为 (流向坐标, 起点距)。"""
        xn = (float(u) - self.cx) / self.fx
        yn = (float(v) - self.cy) / self.fy
        denom = math.sin(self.pitch_rad) + yn * math.cos(self.pitch_rad)
        if abs(denom) < 1e-12:
            raise ValueError("pixel ray is nearly parallel to the water plane")
        ray_scale = self.height_above_water_m / denom
        flow_axis_m = ray_scale * xn
        forward_m = ray_scale * (math.cos(self.pitch_rad) - yn * math.sin(self.pitch_rad))
        station_m = self.camera_dist_m + forward_m
        return flow_axis_m, station_m

    def line_length_m(self, center_x: float, center_y: float, length_px: int) -> float:
        half = float(length_px) / 2.0
        p1 = self.ground_point(center_x - half, center_y)
        p2 = self.ground_point(center_x + half, center_y)
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def hydroview_station_to_camera_y(self, station_m: float) -> float:
        if self.position_to_ref:
            return float(station_m) - self.camera_dist_m
        return float(station_m) + self.camera_dist_m

    def hydroview_camera_y_to_station(self, camera_y_m: float) -> float:
        if self.position_to_ref:
            return self.camera_dist_m + float(camera_y_m)
        return self.camera_dist_m - float(camera_y_m)

    def hydroview_project_y(self, camera_y_m: float, x_m: float = 0.0) -> float:
        if camera_y_m <= 0:
            raise ValueError(f"invalid camera-relative distance: {camera_y_m:.6f} m")
        v = math.tan(math.atan(self.height_above_water_m / camera_y_m) - self.pitch_rad)
        v = v * self.focal_length_mm / self.sensor_size_mm_per_px
        beta = self.pitch_rad + math.atan(v * self.sensor_size_mm_per_px / self.focal_length_mm)
        u = (
            float(x_m)
            * math.sqrt(self.focal_length_mm * self.focal_length_mm + (v * self.sensor_size_mm_per_px) ** 2)
            / self.sensor_size_mm_per_px
            / (self.height_above_water_m / math.sin(beta))
        )
        return (
            math.cos(-self.roll_rad) * v
            - math.sin(-self.roll_rad) * u
            + self.img_h / 2.0
        )

    def ois_dist_y(self, u: float, v: float) -> float:
        j = (
            math.cos(self.roll_rad) * (float(v) - self.img_h / 2.0)
            - math.sin(self.roll_rad) * (float(u) - self.img_w / 2.0)
        )
        beta = self.pitch_rad + math.atan(j * self.sensor_size_mm_per_px / self.focal_length_mm)
        return self.height_above_water_m / math.tan(beta)

    def ois_scale_x(self, v: float) -> float:
        y_inv = self.img_h - float(v) + 1.0
        po = math.sqrt(((self.img_h / 2.0 - y_inv) * self.sensor_size_mm_per_px) ** 2 + self.focal_length_mm ** 2)
        beta_deg = self.pitch_deg + math.degrees(
            math.atan((self.img_h / 2.0 - y_inv) * self.sensor_size_mm_per_px / self.focal_length_mm)
        )
        sin_beta = math.sin(math.radians(beta_deg))
        if abs(sin_beta) < 1e-12:
            raise ValueError(f"invalid OIS scale beta: {beta_deg:.6f} deg")
        return self.height_above_water_m * self.sensor_size_mm_per_px / po / sin_beta

    def hydroview_line_length_m(self, center_y: float, length_px: int) -> float:
        return float(length_px) * self.ois_scale_x(center_y)


def inclusive_range(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("distance step must be positive")
    values: List[float] = []
    n = int(math.floor((end - start) / step + 1e-9))
    for i in range(n + 1):
        values.append(round(start + i * step, 6))
    return values


def line_endpoints(center: Tuple[float, float], length_px: int, angle_deg: float) -> Tuple[Tuple[int, int, int, int], Tuple[float, float]]:
    cx, cy = center
    half = length_px / 2.0
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    x1 = int(round(cx - half * dx))
    y1 = int(round(cy - half * dy))
    x2 = int(round(cx + half * dx))
    y2 = int(round(cy + half * dy))
    return (x1, y1, x2, y2), (dx, dy)


def fixed_arrow_tip_ratio(arrow_len_px: float) -> float:
    """按固定箭头头部大小换算 OpenCV 的 tipLength 比例。"""
    return min(0.45, ARROW_HEAD_PX / max(float(arrow_len_px), 1.0))


def line_fully_inside_frame(center: Tuple[float, float], length_px: int, angle_deg: float, frame_w: int, frame_h: int) -> bool:
    (x1, y1, x2, y2), _ = line_endpoints(center, length_px, angle_deg)
    return 0 <= x1 < frame_w and 0 <= y1 < frame_h and 0 <= x2 < frame_w and 0 <= y2 < frame_h


def normalize_roi(roi: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(v) for v in roi]
    x_min, x_max = sorted((x0, x1))
    y_min, y_max = sorted((y0, y1))
    if x_min < 0 or y_min < 0 or x_max > frame_w or y_max > frame_h:
        raise ValueError(f"ROI is outside frame: {roi}, frame=({frame_w},{frame_h})")
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"invalid ROI: {roi}")
    return x_min, y_min, x_max, y_max


def global_to_local_point(point: Tuple[float, float], roi: Optional[Tuple[int, int, int, int]]) -> Tuple[float, float]:
    if roi is None:
        return float(point[0]), float(point[1])
    x0, y0, _, _ = roi
    return float(point[0] - x0), float(point[1] - y0)


def crop_video_to_roi(video_path: str, out_path: str, roi: Tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = roi
    crop_w, crop_h = x1 - x0, y1 - y0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not fps or math.isnan(fps) or math.isinf(fps):
        fps = 30.0
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (crop_w, crop_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to create ROI video: {out_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame[y0:y1, x0:x1])
    finally:
        cap.release()
        writer.release()


def correct_velocity_px_per_frame(slope_sti: Optional[float], spatial_sample_step: int) -> Optional[float]:
    if slope_sti is None:
        return None
    return float(slope_sti) * float(spatial_sample_step)


def speed_out_of_range(speed: Optional[float]) -> bool:
    if speed is None:
        return False
    abs_speed = abs(float(speed))
    if V_MIN is not None and abs_speed < V_MIN:
        return True
    if V_MAX is not None and abs_speed > V_MAX:
        return True
    return False


def valid_speed_magnitude(row: Dict[str, Any]) -> Optional[float]:
    speed = row.get("speed_m_per_s")
    if speed is None:
        return None
    value = abs(float(speed))
    if not math.isfinite(value) or speed_out_of_range(value):
        return None
    return value


def load_terrain_profile(path: Optional[str]) -> List[Tuple[float, float]]:
    if not path:
        return []
    if not os.path.isfile(path):
        print(f"[警告] 断面地形文件不存在: {path}")
        return []

    rows: List[Tuple[float, float]] = []
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "gbk", "cp936"):
        try:
            with open(path, "r", encoding=encoding) as f:
                rows.clear()
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    parts = text.replace(",", " ").split()
                    if len(parts) < 2:
                        continue
                    rows.append((float(parts[0]), float(parts[1])))
            break
        except Exception as exc:
            last_error = exc
    else:
        print(f"[警告] 断面地形读取失败: {last_error}")
        return []

    rows.sort(key=lambda item: item[0])
    print(f"[断面] 读取地形点 {len(rows)} 个: {os.path.abspath(path)}")
    return rows


def linear_interpolate(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    if abs(x1 - x0) < 1e-12:
        return float(y0)
    return float(y0) + (float(x) - float(x0)) / (float(x1) - float(x0)) * (float(y1) - float(y0))


def terrain_elevation_at(terrain_profile: List[Tuple[float, float]], station_m: float) -> Optional[float]:
    if len(terrain_profile) < 2:
        return None
    for (d0, e0), (d1, e1) in zip(terrain_profile, terrain_profile[1:]):
        if d0 <= station_m <= d1:
            return linear_interpolate(d0, e0, d1, e1, station_m)
    return None


def interpolate_terrain_profile(
    terrain_profile: List[Tuple[float, float]],
    dist_step: float,
) -> Tuple[List[float], List[float], float, float, float]:
    if len(terrain_profile) < 2:
        raise ValueError("Hydroview 布线需要至少 2 个断面地形点")
    if dist_step <= 0:
        raise ValueError("DistStep must be positive")

    section_width = float(terrain_profile[-1][0])
    interp_num = int(math.ceil(section_width / dist_step))
    if interp_num < 2:
        raise ValueError(f"断面宽度或 DistStep 异常: SectionWidth={section_width}, DistStep={dist_step}")

    elevation_max = max(elevation for _, elevation in terrain_profile)
    elevation_min = min(elevation for _, elevation in terrain_profile)
    dist_interp: List[float] = []
    elevation_interp: List[float] = []
    for j in range(interp_num):
        dist = j * dist_step
        elevation = elevation_max
        for (d0, e0), (d1, e1) in zip(terrain_profile, terrain_profile[1:]):
            if d0 <= dist <= d1:
                elevation = linear_interpolate(d0, e0, d1, e1, dist)
                break
        dist_interp.append(float(dist))
        elevation_interp.append(float(elevation))
    return dist_interp, elevation_interp, section_width, elevation_max, elevation_min


def calculate_hydroview_water_layout(
    geom: DongfengCameraGeometry,
    terrain_profile: List[Tuple[float, float]],
    dist_step: float,
) -> Dict[str, Any]:
    dist_interp, elevation_interp, section_width, elevation_max, elevation_min = interpolate_terrain_profile(
        terrain_profile,
        dist_step,
    )
    stage = geom.stage_m

    near_i = None
    for i, elevation in enumerate(elevation_interp):
        if elevation <= stage:
            near_i = i
            break
    if near_i is None or near_i <= 0:
        raise ValueError(f"当前水位 StageInput={stage:.3f} 未在近岸侧与断面地形形成有效水边界")

    far_i = len(elevation_interp) - 1
    while far_i >= 0 and elevation_interp[far_i] > stage:
        far_i -= 1
    if far_i < 0 or far_i + 1 >= len(elevation_interp):
        raise ValueError(f"当前水位 StageInput={stage:.3f} 未在远岸侧与断面地形形成有效水边界")

    near_station = linear_interpolate(
        elevation_interp[near_i - 1],
        dist_interp[near_i - 1],
        elevation_interp[near_i],
        dist_interp[near_i],
        stage,
    )
    far_station = linear_interpolate(
        elevation_interp[far_i],
        dist_interp[far_i],
        elevation_interp[far_i + 1],
        dist_interp[far_i + 1],
        stage,
    )

    if geom.position_to_ref:
        y_near = near_station - geom.camera_dist_m
        y_far = far_station - geom.camera_dist_m
    else:
        y_far = geom.camera_dist_m - near_station
        y_near = geom.camera_dist_m - far_station

    y_far_img = geom.hydroview_project_y(y_far)
    y_near_img = geom.hydroview_project_y(y_near)
    y_bounds = (
        0.0 if y_far_img < 0 else float(math.floor(y_far_img)),
        float(geom.img_h - 1) if y_near_img > geom.img_h - 1 else float(math.ceil(y_near_img)),
    )
    return {
        "dist_interp": dist_interp,
        "elevation_interp": elevation_interp,
        "section_width_m": section_width,
        "elevation_max_m": elevation_max,
        "elevation_min_m": elevation_min,
        "near_station_m": float(near_station),
        "far_station_m": float(far_station),
        "water_width_m": float(abs(far_station - near_station)),
        "y_bounds": y_bounds,
        "y_far_img": float(y_far_img),
        "y_near_img": float(y_near_img),
    }


def hydroview_jp_from_dist_y(geom: DongfengCameraGeometry, dist_y: float) -> float:
    if dist_y <= 0:
        raise ValueError(f"invalid camera-relative distance: {dist_y:.6f} m")
    return (
        math.tan(math.atan(geom.height_above_water_m / dist_y) - geom.pitch_rad)
        * geom.focal_length_mm
        / geom.sensor_size_mm_per_px
    )


def hydroview_calculate_testing_line_centers(
    geom: DongfengCameraGeometry,
    tl_cpx: int,
    dist_step: float,
    y_bounds: Tuple[float, float],
) -> List[Tuple[int, int]]:
    if tl_cpx == 0:
        tl_cpx = geom.img_w // 2
    i0 = (
        math.sin(-geom.roll_rad) * (y_bounds[1] - geom.img_h / 2.0)
        + math.cos(-geom.roll_rad) * (tl_cpx - geom.img_w / 2.0)
        + geom.img_w / 2.0
    )
    j0 = (
        math.cos(-geom.roll_rad) * (y_bounds[1] - geom.img_h / 2.0)
        - math.sin(-geom.roll_rad) * (tl_cpx - geom.img_w / 2.0)
        + geom.img_h / 2.0
    )
    beta0 = geom.pitch_rad + math.atan((j0 - geom.img_h / 2.0) * geom.sensor_size_mm_per_px / geom.focal_length_mm)
    dist_y = geom.height_above_water_m / math.tan(beta0)

    centers: List[Tuple[int, int]] = []
    guard = 0
    while j0 > y_bounds[0]:
        jp = hydroview_jp_from_dist_y(geom, dist_y)
        if y_bounds[0] <= j0 <= y_bounds[1]:
            x = int(
                math.sin(-geom.roll_rad) * (jp - geom.img_h / 2.0)
                + math.cos(-geom.roll_rad) * (i0 - geom.img_w / 2.0)
                + geom.img_w / 2.0
                + 0.5
            )
            y = int(j0 + 0.5)
            centers.append((x, y))
        dist_y += dist_step
        jp = hydroview_jp_from_dist_y(geom, dist_y)
        j0 = math.cos(-geom.roll_rad) * jp - math.sin(-geom.roll_rad) * (i0 - geom.img_w / 2.0) + geom.img_h / 2.0
        guard += 1
        if guard > 10000:
            raise RuntimeError("Hydroview 自动测速线布设迭代过多，请检查 DistStep 和相机参数")
    return centers


def hydroview_calculate_specified_line_centers(
    geom: DongfengCameraGeometry,
    tl_cpx: int,
    station_values: List[float],
    y_bounds: Tuple[float, float],
) -> List[Tuple[int, int]]:
    if tl_cpx == 0:
        tl_cpx = geom.img_w // 2
    i0 = (
        math.sin(-geom.roll_rad) * (y_bounds[1] - geom.img_h / 2.0)
        + math.cos(-geom.roll_rad) * (tl_cpx - geom.img_w / 2.0)
        + geom.img_w / 2.0
    )

    centers: List[Tuple[int, int]] = []
    for station in station_values:
        dist_y = geom.hydroview_station_to_camera_y(station)
        jp = hydroview_jp_from_dist_y(geom, dist_y)
        j0 = math.cos(-geom.roll_rad) * jp - math.sin(-geom.roll_rad) * (i0 - geom.img_w / 2.0) + geom.img_h / 2.0
        if j0 < y_bounds[0]:
            break
        if j0 <= y_bounds[1]:
            x = int(
                math.sin(-geom.roll_rad) * (jp - geom.img_h / 2.0)
                + math.cos(-geom.roll_rad) * (i0 - geom.img_w / 2.0)
                + geom.img_w / 2.0
                + 0.5
            )
            centers.append((x, int(j0 + 0.5)))
    return centers


def run_adaptive_search_quietly(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """运行共用搜索代码，并按需隐藏库里的非 ASCII 进度输出。"""
    if not SUPPRESS_LIBRARY_OUTPUT:
        return _adaptive_direction_search_on_frames(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return _adaptive_direction_search_on_frames(*args, **kwargs)


def build_manual_range_testing_lines(
    cfg: Dict[str, Dict[str, str]],
    frame_w: int,
    frame_h: int,
    length_px: int,
) -> List[Dict[str, Any]]:
    geom = DongfengCameraGeometry(cfg, frame_w, frame_h)
    tl_x = cfg_float(cfg, "MeasuringSettings", "TL_X", 0.0)
    center_x = frame_w / 2.0 if abs(tl_x) < 1e-9 else tl_x
    flow_dir_l2r = cfg_int(cfg, "MeasuringSettings", "FlowDirL2R", 0)

    lines: List[Dict[str, Any]] = []
    for station_m in inclusive_range(DONGFENG_DIST_START_M, DONGFENG_DIST_END_M, DONGFENG_DIST_STEP_M):
        center_y = geom.station_to_pixel_y(station_m)
        center = (float(center_x), float(center_y))
        line_length_m = geom.line_length_m(center[0], center[1], length_px)
        lines.append(
            {
                "station_m": float(station_m),
                "center_x": center[0],
                "center_y": center[1],
                "center_x_px": int(round(center[0])),
                "center_y_px": int(round(center[1])),
                "line_length_px": int(length_px),
                "line_length_m": float(line_length_m),
                "meter_per_pixel_line": float(line_length_m) / float(length_px),
                "flow_dir_l2r": flow_dir_l2r,
            }
        )
    return lines


def build_hydroview_line_row(
    geom: DongfengCameraGeometry,
    source_center_x: float,
    source_center_y: float,
    length_px: int,
    flow_dir_l2r: int,
    layout_info: Dict[str, Any],
    terrain_profile: List[Tuple[float, float]],
    use_fixed_tl: int,
) -> Dict[str, Any]:
    camera_y = geom.ois_dist_y(source_center_x, source_center_y)
    station_m = geom.hydroview_camera_y_to_station(camera_y)
    meter_per_pixel_line = geom.ois_scale_x(source_center_y)
    elevation = terrain_elevation_at(terrain_profile, station_m)
    water_depth_m = None if elevation is None else max(0.0, geom.stage_m - elevation)
    scale_x = geom.frame_w / float(geom.img_w)
    scale_y = geom.frame_h / float(geom.img_h)
    center_x = float(source_center_x) * scale_x
    center_y = float(source_center_y) * scale_y
    y_bounds = layout_info.get("y_bounds", (None, None))
    return {
        "station_m": float(station_m),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "center_x_px": int(round(center_x)),
        "center_y_px": int(round(center_y)),
        "hydroview_center_x": float(source_center_x),
        "hydroview_center_y": float(source_center_y),
        "hydroview_center_x_px": int(round(source_center_x)),
        "hydroview_center_y_px": int(round(source_center_y)),
        "line_length_px": int(length_px),
        "line_length_m": float(length_px) * float(meter_per_pixel_line),
        "meter_per_pixel_line": float(meter_per_pixel_line),
        "flow_dir_l2r": flow_dir_l2r,
        "layout_mode": "hydroview",
        "use_fixed_tl": int(use_fixed_tl),
        "stage_m": float(geom.stage_m),
        "water_depth_m": water_depth_m,
        "water_width_m": layout_info.get("water_width_m"),
        "water_near_station_m": layout_info.get("near_station_m"),
        "water_far_station_m": layout_info.get("far_station_m"),
        "water_y_far_px": None if y_bounds[0] is None else float(y_bounds[0]) * scale_y,
        "water_y_near_px": None if y_bounds[1] is None else float(y_bounds[1]) * scale_y,
        "hydroview_water_y_far_px": y_bounds[0],
        "hydroview_water_y_near_px": y_bounds[1],
    }


def read_fixed_testing_line_stations(cfg: Dict[str, Dict[str, str]]) -> List[float]:
    fixed_num = cfg_int(cfg, "TestingLine", "Num", 0)
    stations: List[float] = []
    for i in range(fixed_num):
        key = f"Dist{i + 1:02d}"
        if key not in cfg.get("TestingLine", {}):
            raise KeyError(f"TestingLine.{key} 不存在，但 Num={fixed_num}")
        stations.append(cfg_float(cfg, "TestingLine", key))
    return stations


def build_hydroview_testing_lines(
    cfg: Dict[str, Dict[str, str]],
    frame_w: int,
    frame_h: int,
    length_px: int,
    terrain_profile: List[Tuple[float, float]],
) -> List[Dict[str, Any]]:
    if not terrain_profile:
        raise ValueError("TESTING_LINE_LAYOUT_MODE='hydroview' 需要 TERRAIN_PROFILE_PATH 指向有效断面地形文件")

    geom = DongfengCameraGeometry(cfg, frame_w, frame_h)
    dist_step = cfg_float(cfg, "MeasuringSettings", "DistStep", DONGFENG_DIST_STEP_M)
    tl_cpx = cfg_int(cfg, "MeasuringSettings", "TL_X", 0)
    flow_dir_l2r = cfg_int(cfg, "MeasuringSettings", "FlowDirL2R", 0)
    use_fixed_tl = cfg_int(cfg, "TestingLine", "UseFixedTL", 0)
    layout_info = calculate_hydroview_water_layout(geom, terrain_profile, dist_step)
    y_bounds = layout_info["y_bounds"]

    if use_fixed_tl:
        fixed_stations = read_fixed_testing_line_stations(cfg)
        centers = hydroview_calculate_specified_line_centers(geom, tl_cpx, fixed_stations, y_bounds)
    else:
        centers = hydroview_calculate_testing_line_centers(geom, tl_cpx, dist_step, y_bounds)

    lines = [
        build_hydroview_line_row(
            geom,
            float(center_x),
            float(center_y),
            length_px,
            flow_dir_l2r,
            layout_info,
            terrain_profile,
            use_fixed_tl,
        )
        for center_x, center_y in centers
    ]
    lines.sort(key=lambda row: row["station_m"])
    print(
        f"[布线] Hydroview: Stage={geom.stage_m:.3f} m, UseFixedTL={use_fixed_tl}, "
        f"水面范围={layout_info['near_station_m']:.3f}..{layout_info['far_station_m']:.3f} m, "
        f"图像y={y_bounds[0]:.1f}..{y_bounds[1]:.1f}, 数量={len(lines)}"
    )
    return lines


def build_dongfeng_testing_lines(
    cfg: Dict[str, Dict[str, str]],
    frame_w: int,
    frame_h: int,
    length_px: int,
    terrain_profile: List[Tuple[float, float]],
) -> List[Dict[str, Any]]:
    mode = TESTING_LINE_LAYOUT_MODE.lower().strip()
    if mode == "manual_range":
        return build_manual_range_testing_lines(cfg, frame_w, frame_h, length_px)
    if mode == "hydroview":
        return build_hydroview_testing_lines(cfg, frame_w, frame_h, length_px, terrain_profile)
    raise ValueError(f"unknown TESTING_LINE_LAYOUT_MODE: {TESTING_LINE_LAYOUT_MODE!r}")


def save_results(outdir: str, results: List[Dict[str, Any]]) -> None:
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "dongfeng_probe_results.csv")
    xlsx_path = os.path.join(outdir, "dongfeng_probe_results.xlsx")
    fieldnames: List[str] = []
    for row in results:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[结果] {os.path.abspath(csv_path)}")

    try:
        import pandas as pd

        pd.DataFrame(results).to_excel(xlsx_path, index=False)
        print(f"[结果] {os.path.abspath(xlsx_path)}")
    except Exception as exc:
        print(f"[警告] Excel 保存失败: {exc}")


def is_single_probe_angle() -> bool:
    return len(inclusive_range(ANGLE_START, ANGLE_END, ANGLE_STEP)) == 1


def write_image(path: str, image: np.ndarray, label: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    encoded.tofile(path)
    print(f"[{label}] {os.path.abspath(path)}")


def to_u8_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)


def draw_hough_on_sti(sti_u8: np.ndarray, best: Dict[str, Any]) -> np.ndarray:
    vis = cv2.cvtColor(sti_u8, cv2.COLOR_GRAY2BGR)
    alpha_deg = best.get("angle")
    h, w = sti_u8.shape[:2]
    if alpha_deg is not None:
        cx, cy = w / 2.0, h / 2.0
        length = float(math.hypot(h, w))
        rad = math.radians(float(alpha_deg))
        ux, uy = math.cos(rad), math.sin(rad)
        p1 = (int(round(cx - length * ux)), int(round(cy - length * uy)))
        p2 = (int(round(cx + length * ux)), int(round(cy + length * uy)))
        cv2.line(vis, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)

    slope = best.get("slope_sti", best.get("slope"))
    peak = best.get("peak_votes", best.get("score"))
    theta_txt = "None" if alpha_deg is None else f"{(float(alpha_deg) - 90.0) % 180.0:.2f}deg"
    line_txt = "None" if alpha_deg is None else f"{float(alpha_deg):.2f}deg"
    slope_txt = "None" if slope is None else f"{float(slope):.4f}"
    peak_txt = "None" if peak is None else f"{float(peak):.0f}"
    text = f"theta_n={theta_txt}, line={line_txt}, slope={slope_txt}, peak={peak_txt}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = max(8, int(round(min(h, w) * 0.016)))
    font_scale = max(0.45, min(h, w) / 900.0)
    thickness = max(1, int(round(font_scale * 2)))
    max_width = max(1, w - 2 * margin)
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    if text_w > max_width:
        font_scale = max(0.25, font_scale * max_width / float(text_w))
        thickness = max(1, int(round(font_scale * 2)))
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    origin = (margin, margin + text_h)
    cv2.rectangle(
        vis,
        (origin[0] - 4, origin[1] - text_h - baseline - 4),
        (origin[0] + text_w + 4, origin[1] + baseline + 4),
        (0, 0, 0),
        -1,
    )
    cv2.putText(vis, text, origin, font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    return vis


def save_single_angle_sti_outputs(temp_dir: str, order: int, best: Dict[str, Any]) -> None:
    raw_sti = best.get("sti_raw")
    if raw_sti is None:
        return

    sti_u8 = to_u8_gray(raw_sti)
    write_image(os.path.join(temp_dir, f"STI{order}.png"), sti_u8, "Temp")

    sobel_mag = build_J1_grad_mag(sti_u8)
    if USE_ROI:
        mask = build_circular_roi_mask(sobel_mag.shape, radius_frac=ROI_RADIUS_FRAC)
        sobel_mag = sobel_mag.copy()
        sobel_mag[~mask] = 0
    write_image(os.path.join(temp_dir, f"STI_sobel{order}.png"), sobel_mag, "Temp")

    hough_vis = draw_hough_on_sti(sti_u8, best)
    write_image(os.path.join(temp_dir, f"STI_MOT{order}.png"), hough_vis, "Temp")


def remove_temp_search_artifacts(temp_dir: str) -> None:
    for name in ("angle_scores.csv", "step8_hough_overlay.png"):
        path = os.path.join(temp_dir, name)
        if os.path.exists(path):
            os.remove(path)


def read_gray_first_frame(video_path: str) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def draw_testing_lines(frame: np.ndarray, results: List[Dict[str, Any]], length_px: int) -> None:
    for row in results:
        point = (int(row["point_x"]), int(row["point_y"]))
        angle_value = row.get("angle_probe_deg")
        angle = float(ANGLE_START if angle_value is None else angle_value)
        (x1, y1, x2, y2), _ = line_endpoints(point, length_px, angle)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 2, cv2.LINE_AA)


def save_dongfeng_overlays(video_path: str, outdir: str, results: List[Dict[str, Any]], length_px: int) -> None:
    if not results:
        return
    base = read_gray_first_frame(video_path)
    if base is None:
        print("[叠加图] 无法读取首帧")
        return

    line_frame = base.copy()
    draw_testing_lines(line_frame, results, length_px)
    layout_path = os.path.join(outdir, "dongfeng_world_layout_overview.png")
    write_image(layout_path, line_frame, "叠加图")

    velocity_frame = base.copy()
    draw_testing_lines(velocity_frame, results, length_px)
    valid_speeds = [speed for row in results if (speed := valid_speed_magnitude(row)) is not None]
    max_speed = max(valid_speeds) if valid_speeds else None
    min_arrow_len = max(20, int(round(length_px * 0.08)))
    max_arrow_len = max(min_arrow_len + 1, int(round(length_px * 0.25)))

    for row in results:
        speed = valid_speed_magnitude(row)
        if speed is None or not max_speed:
            continue
        point = (int(row["point_x"]), int(row["point_y"]))
        angle_value = row.get("angle_probe_deg")
        angle = float(ANGLE_START if angle_value is None else angle_value)
        _, direction = line_endpoints(point, length_px, angle)
        velocity = row.get("velocity_mps")
        sign = 1 if velocity is None or float(velocity) >= 0 else -1
        arrow_len = int(round(min_arrow_len + speed / max_speed * (max_arrow_len - min_arrow_len)))
        dx, dy = direction
        end = (int(round(point[0] + sign * dx * arrow_len)), int(round(point[1] + sign * dy * arrow_len)))
        cv2.arrowedLine(
            velocity_frame,
            point,
            end,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
            tipLength=fixed_arrow_tip_ratio(arrow_len),
        )

    velocity_path = os.path.join(outdir, "dongfeng_world_velocity_overview.png")
    write_image(velocity_path, velocity_frame, "叠加图")


def save_surface_velocity_profile(
    temp_dir: str,
    results: List[Dict[str, Any]],
    terrain_profile: List[Tuple[float, float]],
) -> None:
    if not results:
        return
    width, height = 900, 320
    left, right, top, bottom = 72, 24, 24, 42
    axis_y = 150
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    stations = [float(row["station_m"]) for row in results]
    speeds = [speed for row in results if (speed := valid_speed_magnitude(row)) is not None]
    if not stations:
        return
    terrain_stations = [point[0] for point in terrain_profile]
    x_min = 0.0
    x_max = max(stations + terrain_stations)
    max_speed = max(speeds) if speeds else 1.0
    max_speed = max(max_speed, 1e-9)
    plot_w = width - left - right
    speed_h = axis_y - top - 10
    depth_h = height - bottom - axis_y - 16
    terrain_depths: List[Tuple[float, float]] = []
    max_depth = 0.0

    cv2.line(canvas, (left, axis_y), (width - right, axis_y), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, (left, top), (left, height - bottom), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "V", (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "B", (12, 226), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    tick_font = cv2.FONT_HERSHEY_SIMPLEX
    tick_font_scale = 0.42
    tick_thickness = 1
    for value in np.linspace(0.0, max_speed, 4):
        y = int(round(axis_y - float(value) / max_speed * speed_h))
        cv2.line(canvas, (left - 5, y), (left, y), (0, 0, 0), 1, cv2.LINE_AA)
        if abs(float(value)) < 1e-12:
            continue
        label = f"{value:.1f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, tick_font, tick_font_scale, tick_thickness)
        cv2.putText(
            canvas,
            label,
            (left - text_w - 8, y + text_h // 2),
            tick_font,
            tick_font_scale,
            (0, 0, 0),
            tick_thickness,
            cv2.LINE_AA,
        )

    for tick in np.linspace(x_min, x_max, 6):
        x = int(round(left + (tick - x_min) / max(x_max - x_min, 1e-9) * plot_w))
        cv2.line(canvas, (x, axis_y - 4), (x, axis_y + 4), (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{tick:.1f}", (x - 16, axis_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    if terrain_profile:
        stage_values = [float(row["stage_m"]) for row in results if row.get("stage_m") is not None]
        water_level = stage_values[0] if stage_values else max(elevation for _, elevation in terrain_profile)
        terrain_depths = [(station, max(0.0, water_level - elevation)) for station, elevation in terrain_profile]
        max_depth = max((depth for _, depth in terrain_depths), default=0.0)
        if max_depth > 0:
            for value in np.linspace(0.0, max_depth, 5)[1:]:
                y = int(round(axis_y + float(value) / max_depth * depth_h))
                cv2.line(canvas, (left - 5, y), (left, y), (0, 0, 0), 1, cv2.LINE_AA)
                label = f"{value:.1f}"
                (text_w, text_h), baseline = cv2.getTextSize(label, tick_font, tick_font_scale, tick_thickness)
                cv2.putText(
                    canvas,
                    label,
                    (left - text_w - 8, y + text_h // 2),
                    tick_font,
                    tick_font_scale,
                    (0, 0, 0),
                    tick_thickness,
                    cv2.LINE_AA,
                )
            points = []
            for station, depth in terrain_depths:
                x = int(round(left + (station - x_min) / max(x_max - x_min, 1e-9) * plot_w))
                y = int(round(axis_y + depth / max_depth * depth_h))
                points.append((x, y))
            for p1, p2 in zip(points, points[1:]):
                cv2.line(canvas, p1, p2, (255, 0, 0), 2, cv2.LINE_AA)

    for row in results:
        speed = valid_speed_magnitude(row)
        if speed is None:
            continue
        station = float(row["station_m"])
        x = int(round(left + (station - x_min) / max(x_max - x_min, 1e-9) * plot_w))
        y_top = int(round(axis_y - speed / max_speed * speed_h))
        cv2.line(canvas, (x, axis_y), (x, y_top), (0, 230, 0), 3, cv2.LINE_AA)

    out_path = os.path.join(temp_dir, "cross_section_velocity_profile.png")
    write_image(out_path, canvas, "断面图")


def main() -> None:
    configure_output_encoding()
    if not os.path.isfile(VIDEO):
        raise FileNotFoundError(f"视频不存在: {VIDEO}")
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(f"配置文件不存在: {CONFIG_PATH}")

    cfg = load_config_ini(CONFIG_PATH)
    outdir = init_debug_dir(tag="dongfeng-world-layout")
    process_video = VIDEO
    roi_box: Optional[Tuple[int, int, int, int]] = None
    roi_offset = (0, 0)

    cap0 = cv2.VideoCapture(VIDEO)
    ok0, frame0 = cap0.read()
    cap0.release()
    if not ok0:
        raise RuntimeError(f"无法读取视频首帧: {VIDEO}")
    frame_h, frame_w = frame0.shape[:2]
    terrain_profile = load_terrain_profile(TERRAIN_PROFILE_PATH)

    testing_lines_global = build_dongfeng_testing_lines(cfg, frame_w, frame_h, LENGTH_PX, terrain_profile)
    if not testing_lines_global:
        raise RuntimeError("未生成任何测速线，请检查布设模式、水位、断面地形和相机参数")
    if TESTING_LINE_LAYOUT_MODE.lower().strip() == "manual_range":
        expected_count = int(round((DONGFENG_DIST_END_M - DONGFENG_DIST_START_M) / DONGFENG_DIST_STEP_M)) + 1
        if len(testing_lines_global) != expected_count:
            raise RuntimeError(f"unexpected testing-line count: {len(testing_lines_global)} != {expected_count}")

    if ROI is not None:
        roi_box = normalize_roi(ROI, frame_w, frame_h)
        roi_offset = (roi_box[0], roi_box[1])
        roi_video_path = os.path.join(outdir, "roi_cropped.mp4")
        crop_video_to_roi(VIDEO, roi_video_path, roi_box)
        process_video = roi_video_path
        check_w = roi_box[2] - roi_box[0]
        check_h = roi_box[3] - roi_box[1]
    else:
        check_w = frame_w
        check_h = frame_h

    spatial_count = spatial_sample_count(LENGTH_PX, SPATIAL_SAMPLE_STEP)
    print(f"[输出] 输出目录: {outdir}")
    if TESTING_LINE_LAYOUT_MODE.lower().strip() == "manual_range":
        print(
            f"[配置] 手动范围布线: 起点距={DONGFENG_DIST_START_M:.1f}..{DONGFENG_DIST_END_M:.1f} m, "
            f"间隔={DONGFENG_DIST_STEP_M:.1f} m, 数量={len(testing_lines_global)}"
        )
    else:
        print(f"[配置] Hydroview 源码布线: 数量={len(testing_lines_global)}")
    print(
        f"[配置] LENGTH_PX={LENGTH_PX}, MAX_FRAMES={MAX_FRAMES}, "
        f"SPATIAL_SAMPLE_STEP={SPATIAL_SAMPLE_STEP}, spatial_sample_count={spatial_count}"
    )
    print(
        f"[配置] VOTE_THETA_RANGE={VOTE_THETA_RANGE}, coarse_to_fine={USE_COARSE_TO_FINE_THETA}, "
        f"coarse_res={VOTE_THETA_COARSE_RES_DEG}, fine_res={VOTE_THETA_FINE_RES_DEG}, "
        f"fine_half_width={VOTE_THETA_FINE_HALF_WIDTH_DEG}"
    )

    frames, video_fps = _load_video_frames(
        process_video,
        MAX_FRAMES,
        start_frame=START_FRAME,
        start_time_sec=START_TIME_SEC,
    )
    effective_fps = float(FPS) if FPS is not None else float(video_fps)
    print(f"[帧率] {effective_fps:.6f}")

    single_angle_mode = is_single_probe_angle()
    temp_dir = os.path.join(outdir, TEMP_DIR_NAME)
    if single_angle_mode:
        os.makedirs(temp_dir, exist_ok=True)
        print(f"[输出] 单角度 STI 输出目录: {os.path.abspath(temp_dir)}")

    results: List[Dict[str, Any]] = []
    for idx, line in enumerate(testing_lines_global):
        center_global = (line["center_x"], line["center_y"])
        center_local = global_to_local_point(center_global, roi_box)
        for angle in inclusive_range(ANGLE_START, ANGLE_END, ANGLE_STEP):
            if not line_fully_inside_frame(center_local, LENGTH_PX, angle, check_w, check_h):
                raise ValueError(
                    f"测速线超出图像/ROI: 起点距={line['station_m']:.3f}, "
                    f"中心={center_global}, 局部中心={center_local}, angle={angle}, LENGTH_PX={LENGTH_PX}"
                )

        station_tag = f"station_{line['station_m']:.1f}m".replace(".", "p")
        suffix = f"{idx:02d}_{station_tag}_x{line['center_x_px']}_y{line['center_y_px']}"
        debug_suffix = TEMP_DIR_NAME if single_angle_mode else suffix
        with push_debug_dir(debug_suffix):
            best = run_adaptive_search_quietly(
                frames,
                video_fps,
                center_local,
                LENGTH_PX,
                ANGLE_START,
                ANGLE_END,
                ANGLE_STEP,
                spatial_sample_step=SPATIAL_SAMPLE_STEP,
                max_frames=MAX_FRAMES,
                use_circular_roi=USE_ROI,
                roi_radius_frac=ROI_RADIUS_FRAC,
                edge_method=EDGE_METHOD,
                direction_method=DIRECTION_METHOD,
                verbose=VERBOSE,
                use_E_asym=USE_E_ASYM,
                use_M_mono=USE_M_MONO,
                m_mono_peak_ratio=M_MONO_PEAK_RATIO,
                vote_theta_res_deg=VOTE_THETA_RES_DEG,
                vote_k_ratio=VOTE_K_RATIO,
                vote_theta_range=VOTE_THETA_RANGE,
                use_coarse_to_fine_theta=USE_COARSE_TO_FINE_THETA,
                vote_theta_coarse_res_deg=VOTE_THETA_COARSE_RES_DEG,
                vote_theta_fine_res_deg=VOTE_THETA_FINE_RES_DEG,
                vote_theta_fine_half_width_deg=VOTE_THETA_FINE_HALF_WIDTH_DEG,
                k_sigma=K_SIGMA,
                save_candidate_overlays=False,
                save_all_sti=False if single_angle_mode else SAVE_ALL_STI,
                top_k_candidates=TOP_K_CANDIDATES,
                score_mode=SCORE_MODE,
                save_debug_images=False if single_angle_mode else SAVE_DEBUG_IMAGES,
            )
            raw_sti = best.get("sti_raw")
            if raw_sti is not None and not single_angle_mode:
                h, w = raw_sti.shape[:2]
                _save_img(
                    f"{suffix}_STI_step{SPATIAL_SAMPLE_STEP}_len{LENGTH_PX}_size{w}x{h}.png",
                    raw_sti,
                )
        if single_angle_mode:
            if best.get("sti_raw") is None:
                fallback_angle = inclusive_range(ANGLE_START, ANGLE_END, ANGLE_STEP)[0]
                fallback_sti = build_sti_from_frames(
                    frames,
                    (int(round(center_local[0])), int(round(center_local[1]))),
                    LENGTH_PX,
                    fallback_angle,
                    spatial_sample_step=SPATIAL_SAMPLE_STEP,
                )
                if fallback_sti is not None:
                    best["sti_raw"] = fallback_sti
                    if best.get("angle_probe") is None:
                        best["angle_probe"] = fallback_angle
            remove_temp_search_artifacts(temp_dir)
            save_single_angle_sti_outputs(temp_dir, idx + 1, best)

        best["fps"] = effective_fps
        slope_sti = best.get("slope_sti", best.get("slope"))
        velocity_px_per_frame = best.get("velocity_px_per_frame_corrected")
        if velocity_px_per_frame is None:
            velocity_px_per_frame = correct_velocity_px_per_frame(slope_sti, SPATIAL_SAMPLE_STEP)
        velocity_mps = None
        speed_m_per_s = None
        if velocity_px_per_frame is not None:
            velocity_mps = float(velocity_px_per_frame) * float(line["meter_per_pixel_line"]) * effective_fps
            speed_m_per_s = abs(velocity_mps)

        result_row: Dict[str, Any] = {
            "index": idx,
            "station_m": line["station_m"],
            "point_x": line["center_x_px"],
            "point_y": line["center_y_px"],
            "point_x_float": line["center_x"],
            "point_y_float": line["center_y"],
            "hydroview_point_x": line.get("hydroview_center_x_px"),
            "hydroview_point_y": line.get("hydroview_center_y_px"),
            "hydroview_point_x_float": line.get("hydroview_center_x"),
            "hydroview_point_y_float": line.get("hydroview_center_y"),
            "point_x_local": center_local[0],
            "point_y_local": center_local[1],
            "angle_probe_deg": best.get("angle_probe"),
            "alpha_deg": best.get("angle"),
            "score": best.get("score"),
            "peak_votes": best.get("peak_votes"),
            "peak_ratio": best.get("peak_ratio"),
            "length_px": LENGTH_PX,
            "line_length_px": line["line_length_px"],
            "line_length_m": line["line_length_m"],
            "meter_per_pixel_line": line["meter_per_pixel_line"],
            "spatial_sample_step": SPATIAL_SAMPLE_STEP,
            "spatial_sample_count": spatial_count,
            "max_frames": MAX_FRAMES,
            "slope_sti": slope_sti,
            "velocity_px_per_frame_corrected": velocity_px_per_frame,
            "velocity_mps": velocity_mps,
            "speed_m_per_s": speed_m_per_s,
            "fps": effective_fps,
            "flow_dir_l2r": line["flow_dir_l2r"],
            "layout_mode": line.get("layout_mode", TESTING_LINE_LAYOUT_MODE),
            "use_fixed_tl": line.get("use_fixed_tl"),
            "stage_m": line.get("stage_m"),
            "water_depth_m": line.get("water_depth_m"),
            "water_width_m": line.get("water_width_m"),
            "water_near_station_m": line.get("water_near_station_m"),
            "water_far_station_m": line.get("water_far_station_m"),
            "water_y_far_px": line.get("water_y_far_px"),
            "water_y_near_px": line.get("water_y_near_px"),
            "hydroview_water_y_far_px": line.get("hydroview_water_y_far_px"),
            "hydroview_water_y_near_px": line.get("hydroview_water_y_near_px"),
        }
        if USE_E_ASYM:
            result_row["E_asym"] = best.get("E_asym")
        if USE_M_MONO:
            result_row["M_mono"] = best.get("M_mono")
        results.append(result_row)

        speed_txt = "N/A" if speed_m_per_s is None else f"{speed_m_per_s:.4f} m/s"
        print(
            f"[东风渠] #{idx:02d} 起点距={line['station_m']:.1f} m "
            f"中心=({line['center_x_px']},{line['center_y_px']}) "
            f"线长={line['line_length_m']:.3f} m m/px={line['meter_per_pixel_line']:.6f} "
            f"速度={speed_txt}"
        )

    save_results(outdir, results)

    overlay_rows = results
    if roi_box is not None:
        ox, oy = roi_offset
        overlay_rows = []
        for row in results:
            local_row = dict(row)
            local_row["point_x"] = int(round(float(row["point_x"]) - ox))
            local_row["point_y"] = int(round(float(row["point_y"]) - oy))
            overlay_rows.append(local_row)
    save_dongfeng_overlays(process_video, outdir, overlay_rows, LENGTH_PX)
    save_surface_velocity_profile(temp_dir, results, terrain_profile)

    print("[完成] 东风渠现实等距测速线处理完成。")


if __name__ == "__main__":
    main()
    t1 = time.perf_counter()
    print(f"[耗时] total = {t1 - t0:.3f} s")
