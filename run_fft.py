# -*- coding: utf-8 -*-
import csv
import math
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


t0 = time.perf_counter()

# ========== 用户配置区（按需修改） ==========
VIDEO = r"D:\Desktop\东风渠\测流视频\20260529\110000_undistort.mp4"
CENTER: Tuple[int, int] = (1919, 1200)  # 手动中心点（像素坐标）

# 多点测速参数
USE_BATCH_LINE_PROBING = False
BANK_POINT: Tuple[int, int] = (1837, 193)  # 岸边点（与 CENTER 组成测速直线）
PROBE_INTERVAL_PX = 200

# STI 测线参数（角度搜索范围：线方向）
LENGTH_PX = 1500
ANGLE_START, ANGLE_END, ANGLE_STEP = 0, 0, 1
MAX_FRAMES = 750
SPATIAL_SAMPLE_STEP = 2

# 从某帧或某秒开始
START_FRAME: Optional[int] = None
START_TIME_SEC: Optional[float] = None

# 最佳测速线方向选择策略：peak_votes / peak_ratio
SCORE_MODE: str = "peak_ratio"

USE_ROI = True
ROI_RADIUS_FRAC: float = 0.9
ROI: Optional[Tuple[int, int, int, int]] = None  # 矩形 ROI: (x0, y0, x1, y1)，None 表示关闭
VERBOSE = True

# 单点测速：保存所有 STI 中间结果（按步骤分文件夹）
SAVE_ALL_STI: bool = True
SAVE_DEBUG_IMAGES: bool = True

# 速度阈值设置（m/s），可按需修改；留 None 表示不限制
V_MIN: Optional[float] = None
V_MAX: Optional[float] = None

# 帧率（建议手动给准值；留 None 则使用视频元数据）
FPS: Optional[float] = None

# 比例尺：二选一
SCALE_M_PER_PIXEL: Optional[float] = None
CALIB_REAL_M: Optional[float] = None
CALIB_LINE_XYXY: Optional[Tuple[int, int, int, int]] = (476, 835, 3356, 809)

# FFT 角度谱参数
FFT_ANGLE_RES_DEG: float = 1.0
FFT_MIN_RADIUS: float = 3.0
FFT_MAX_RADIUS_FRAC: float = 0.95
FFT_USE_HANN_WINDOW: bool = True
FFT_LOG_MAGNITUDE: bool = True
FFT_SMOOTH_WINDOW: int = 5
# 手动输入曲线图上标注的左右边界角度；两者都不为 None 时也作为峰值搜索范围
FFT_LEFT_BOUND_DEG: Optional[float] = 80
FFT_RIGHT_BOUND_DEG: Optional[float] = 89
FFT_PEAK_SEARCH_RANGE_DEG: Tuple[float, float] = (0.0, 180.0)  # 左右边界未完整设置时使用
# ==========================================


DEBUG_RUN_DIR: Optional[str] = None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def init_debug_dir(base: str = "out", tag: str = "") -> str:
    global DEBUG_RUN_DIR
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}{('-' + tag) if tag else ''}"
    DEBUG_RUN_DIR = os.path.join(base, name)
    _ensure_dir(DEBUG_RUN_DIR)
    return DEBUG_RUN_DIR


@contextmanager
def push_debug_dir(suffix: str):
    global DEBUG_RUN_DIR
    prev = DEBUG_RUN_DIR
    subdir = None
    if DEBUG_RUN_DIR:
        subdir = os.path.join(DEBUG_RUN_DIR, suffix)
        _ensure_dir(subdir)
        DEBUG_RUN_DIR = subdir
    try:
        yield subdir
    finally:
        DEBUG_RUN_DIR = prev


def _save_img(name: str, img: np.ndarray) -> str:
    if DEBUG_RUN_DIR is None:
        init_debug_dir()
    path = os.path.join(DEBUG_RUN_DIR or ".", name)
    _ensure_dir(os.path.dirname(path))

    if img.dtype in (np.float32, np.float64):
        mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
            vis = np.zeros_like(img, dtype=np.uint8)
        else:
            vis = np.clip((img - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)
    elif img.dtype == np.uint16:
        vis = np.clip(img / 256.0, 0, 255).astype(np.uint8)
    else:
        vis = img

    cv2.imwrite(path, vis)
    print(f"[save] {os.path.abspath(path)}")
    return path


def spatial_sample_count(length_px: int, spatial_sample_step: int = 1) -> int:
    length_px = int(length_px)
    spatial_sample_step = int(spatial_sample_step)
    if length_px <= 0:
        raise ValueError("LENGTH_PX 必须为正整数")
    if spatial_sample_step <= 0:
        raise ValueError("SPATIAL_SAMPLE_STEP 必须为正整数")
    if length_px % spatial_sample_step != 0:
        raise ValueError(
            f"LENGTH_PX={length_px} 不能被 SPATIAL_SAMPLE_STEP={spatial_sample_step} 整除"
        )
    return length_px // spatial_sample_step


def _line_sample_maps(
    center: Tuple[int, int],
    length_px: int,
    angle_deg: float,
    spatial_sample_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    cx, cy = center
    half = length_px / 2.0
    theta = math.radians(angle_deg)
    dx = math.cos(theta)
    dy = math.sin(theta)
    sample_count = spatial_sample_count(length_px, spatial_sample_step)

    if int(spatial_sample_step) == 1:
        xs = np.linspace(cx - half * dx, cx + half * dx, num=length_px, dtype=np.float32)
        ys = np.linspace(cy - half * dy, cy + half * dy, num=length_px, dtype=np.float32)
    else:
        offsets = (
            np.arange(sample_count, dtype=np.float32) - sample_count / 2.0
        ) * float(spatial_sample_step)
        xs = (cx + offsets * dx).astype(np.float32)
        ys = (cy + offsets * dy).astype(np.float32)

    return xs.reshape(1, sample_count), ys.reshape(1, sample_count)


def build_sti_from_frames(
    frames_gray: List[np.ndarray],
    center: Tuple[int, int],
    length_px: int,
    angle_deg: float,
    spatial_sample_step: int = 1,
) -> Optional[np.ndarray]:
    if not frames_gray:
        return None
    map_x, map_y = _line_sample_maps(center, length_px, angle_deg, spatial_sample_step)
    rows = []
    for gray in frames_gray:
        row = cv2.remap(
            gray,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        rows.append(row)
    sti = np.vstack(rows)
    return np.clip(sti, 0, 255).astype(np.uint8)


def compute_scale_from_first_frame(
    video_path: str,
    xyxy: Tuple[int, int, int, int],
    real_meters: float,
) -> float:
    cap = cv2.VideoCapture(video_path)
    ok, _ = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("无法读取视频首帧用于标定")
    x1, y1, x2, y2 = xyxy
    px = math.hypot(x2 - x1, y2 - y1)
    if px < 1:
        raise RuntimeError("标定两点太近或坐标不正确")
    m_per_px = real_meters / px
    print(f"[calib] 像素距离={px:.2f}px, 真实距离={real_meters:.3f}m -> {m_per_px:.6f} m/px")
    return m_per_px


def _normalize_roi(roi: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(v) for v in roi]
    x_min, x_max = sorted((x0, x1))
    y_min, y_max = sorted((y0, y1))
    if x_min < 0 or y_min < 0 or x_max > frame_w or y_max > frame_h:
        raise ValueError(f"ROI 超出视频边界: {roi}, frame=({frame_w},{frame_h})")
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"ROI 无效: {roi}")
    return x_min, y_min, x_max, y_max


def _point_in_roi(point: Tuple[int, int], roi: Tuple[int, int, int, int]) -> bool:
    x, y = int(point[0]), int(point[1])
    x0, y0, x1, y1 = roi
    return x0 <= x < x1 and y0 <= y < y1


def _global_to_local_point(point: Tuple[int, int], roi: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int]:
    if roi is None:
        return int(point[0]), int(point[1])
    x0, y0, _, _ = roi
    return int(point[0] - x0), int(point[1] - y0)


def _global_to_local_line(
    line_xyxy: Optional[Tuple[int, int, int, int]],
    roi: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    if line_xyxy is None:
        return None
    if roi is None:
        return tuple(int(v) for v in line_xyxy)
    x0, y0, _, _ = roi
    x1, y1, x2, y2 = line_xyxy
    return int(x1 - x0), int(y1 - y0), int(x2 - x0), int(y2 - y0)


def _iter_angles(angle_start: float, angle_end: float, angle_step: float):
    if angle_step == 0:
        raise ValueError("ANGLE_STEP 不能为 0")
    if angle_step > 0:
        a = angle_start
        while a <= angle_end + 1e-9:
            yield float(a)
            a += angle_step
    else:
        a = angle_start
        while a >= angle_end - 1e-9:
            yield float(a)
            a += angle_step


def _line_endpoints(center: Tuple[int, int], length_px: int, angle_deg: float):
    cx, cy = center
    half = length_px / 2.0
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    x1 = int(round(cx - half * dx))
    y1 = int(round(cy - half * dy))
    x2 = int(round(cx + half * dx))
    y2 = int(round(cy + half * dy))
    return (x1, y1, x2, y2), (dx, dy)


def _line_fully_inside_frame(
    center: Tuple[int, int],
    length_px: int,
    angle_deg: float,
    frame_w: int,
    frame_h: int,
) -> bool:
    (x1, y1, x2, y2), _ = _line_endpoints(center, length_px, angle_deg)
    return 0 <= x1 < frame_w and 0 <= y1 < frame_h and 0 <= x2 < frame_w and 0 <= y2 < frame_h


def _is_speed_out_of_range(speed: Optional[float]) -> bool:
    if speed is None:
        return False
    abs_speed = abs(speed)
    if V_MIN is not None and abs_speed < V_MIN:
        return True
    if V_MAX is not None and abs_speed > V_MAX:
        return True
    return False


def _correct_velocity_px_per_frame(
    slope_sti: Optional[float],
    spatial_sample_step: int,
) -> Optional[float]:
    if slope_sti is None:
        return None
    return float(slope_sti) * float(spatial_sample_step)


def _velocity_mps(
    velocity_px_per_frame: Optional[float],
    m_per_px: Optional[float],
    fps: Optional[float],
) -> Optional[float]:
    if velocity_px_per_frame is None or m_per_px is None or fps is None:
        return None
    return float(velocity_px_per_frame) * float(m_per_px) * float(fps)


def _load_video_frames(
    video_path: str,
    max_frames: int,
    start_frame: Optional[int] = None,
    start_time_sec: Optional[float] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not fps or math.isnan(fps) or math.isinf(fps):
        fps = 30.0

    target_frame = 0
    if start_time_sec is not None:
        if start_time_sec < 0:
            cap.release()
            raise ValueError("START_TIME_SEC 不能为负数")
        target_frame = int(round(float(start_time_sec) * fps))
    elif start_frame is not None:
        if start_frame < 0:
            cap.release()
            raise ValueError("START_FRAME 不能为负数")
        target_frame = int(start_frame)

    if target_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    frames: List[np.ndarray] = []
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames > 0 and count >= max_frames):
            break
        if roi is not None:
            x0, y0, x1, y1 = roi
            frame = frame[y0:y1, x0:x1]
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        count += 1

    cap.release()
    if not frames:
        raise RuntimeError("读取到 0 帧")
    return frames, fps


def _read_first_frame(
    video_path: str,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"无法读取视频首帧: {video_path}")
    if roi is not None:
        x0, y0, x1, y1 = roi
        frame = frame[y0:y1, x0:x1]
    return frame


def _circular_smooth(values: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1 or values.size == 0:
        return values.astype(np.float64, copy=True)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values.astype(np.float64, copy=False), (pad, pad), mode="wrap")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _angle_in_range(theta: np.ndarray, angle_range: Tuple[float, float]) -> np.ndarray:
    lo = float(angle_range[0]) % 180.0
    hi = float(angle_range[1]) % 180.0
    if abs((hi - lo) % 180.0) < 1e-9 and abs(float(angle_range[1]) - float(angle_range[0])) >= 180.0:
        return np.ones(theta.shape, dtype=bool)
    if lo <= hi:
        return (theta >= lo) & (theta <= hi)
    return (theta >= lo) | (theta <= hi)


def _normalize_angle_marker(angle: Optional[float]) -> Optional[float]:
    if angle is None:
        return None
    value = float(angle) % 180.0
    if abs(value - 180.0) < 1e-9:
        value = 0.0
    return value


def _effective_fft_peak_range() -> Tuple[float, float]:
    if FFT_LEFT_BOUND_DEG is not None and FFT_RIGHT_BOUND_DEG is not None:
        return float(FFT_LEFT_BOUND_DEG), float(FFT_RIGHT_BOUND_DEG)
    return FFT_PEAK_SEARCH_RANGE_DEG


def compute_fft_angle_response(sti_u8: np.ndarray) -> Dict[str, Any]:
    if sti_u8.ndim != 2:
        raise ValueError("STI 必须是单通道灰度图")

    h, w = sti_u8.shape[:2]
    x = sti_u8.astype(np.float64)
    x -= float(np.mean(x))

    if FFT_USE_HANN_WINDOW and h > 1 and w > 1:
        wy = np.hanning(h).reshape(h, 1)
        wx = np.hanning(w).reshape(1, w)
        x = x * wy * wx

    fft = np.fft.fftshift(np.fft.fft2(x))
    mag = np.abs(fft)
    mag_for_response = np.log1p(mag) if FFT_LOG_MAGNITUDE else mag

    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    dx = xx.astype(np.float64) - float(cx)
    dy = yy.astype(np.float64) - float(cy)
    radius = np.sqrt(dx * dx + dy * dy)

    max_radius = max(FFT_MIN_RADIUS + 1.0, float(min(cx, cy)) * float(FFT_MAX_RADIUS_FRAC))
    ring = (radius >= float(FFT_MIN_RADIUS)) & (radius <= max_radius)

    theta = np.degrees(np.arctan2(dy, dx))
    theta = np.mod(theta, 180.0)

    angle_res = float(FFT_ANGLE_RES_DEG)
    if angle_res <= 0:
        raise ValueError("FFT_ANGLE_RES_DEG 必须为正数")
    bin_count = max(1, int(round(180.0 / angle_res)))
    angle_res = 180.0 / float(bin_count)
    bin_idx_full = np.floor(theta / angle_res).astype(np.int32)
    bin_idx_full = np.clip(bin_idx_full, 0, bin_count - 1)

    vals = mag_for_response[ring].astype(np.float64, copy=False).ravel()
    bin_idx = bin_idx_full[ring].ravel()
    sum_per_bin = np.bincount(bin_idx, weights=vals, minlength=bin_count)
    cnt_per_bin = np.bincount(bin_idx, minlength=bin_count)
    response = sum_per_bin / np.maximum(cnt_per_bin, 1)
    response = _circular_smooth(response, FFT_SMOOTH_WINDOW)

    theta_axis = (np.arange(bin_count, dtype=np.float64) + 0.5) * angle_res
    valid = _angle_in_range(theta_axis, _effective_fft_peak_range())
    if not np.any(valid):
        valid = np.ones(theta_axis.shape, dtype=bool)

    masked_response = response.copy()
    masked_response[~valid] = -np.inf
    peak_idx = int(np.argmax(masked_response))
    peak_angle = float(theta_axis[peak_idx])
    peak_value = float(response[peak_idx])
    response_sum = float(np.sum(np.maximum(response[valid], 0.0)))
    peak_ratio = peak_value / (response_sum + 1e-12)

    line_angle = (peak_angle + 90.0) % 180.0
    tan_a = math.tan(math.radians(line_angle))
    slope = None if abs(tan_a) < 1e-9 else 1.0 / tan_a

    return {
        "theta_deg": theta_axis,
        "response": response,
        "fft_magnitude": mag_for_response,
        "peak_idx": peak_idx,
        "fft_peak_angle_deg": peak_angle,
        "fft_peak_value": peak_value,
        "fft_peak_ratio": peak_ratio,
        "angle": line_angle,
        "slope": slope,
        "score": peak_ratio if SCORE_MODE == "peak_ratio" else peak_value,
        "fft_min_radius": float(FFT_MIN_RADIUS),
        "fft_max_radius": float(max_radius),
    }


def _curve_point(
    angle_deg: float,
    value: float,
    x0: int,
    x1: int,
    y_top: int,
    y_axis: int,
    v_min: float,
    v_max: float,
) -> Tuple[int, int]:
    x = int(round(x0 + (float(angle_deg) / 180.0) * (x1 - x0)))
    if v_max - v_min < 1e-12:
        y = y_axis
    else:
        norm = (float(value) - v_min) / (v_max - v_min)
        y = int(round(y_axis - np.clip(norm, 0.0, 1.0) * (y_axis - y_top)))
    return x, y


def _put_label(
    canvas: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.75,
    thickness: int = 2,
) -> None:
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def save_fft_curve(
    theta_deg: np.ndarray,
    response: np.ndarray,
    peak_angle_deg: float,
    left_bound_deg: Optional[float],
    right_bound_deg: Optional[float],
    save_name: str = "fft_curve.png",
) -> str:
    width, height = 1800, 300
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, x1 = 12, width - 12
    y_top, y_axis = 22, height - 30

    finite = response[np.isfinite(response)]
    if finite.size == 0:
        v_min, v_max = 0.0, 1.0
    else:
        v_min, v_max = float(np.min(finite)), float(np.max(finite))

    points = [
        _curve_point(float(a), float(v), x0, x1, y_top, y_axis, v_min, v_max)
        for a, v in zip(theta_deg, response)
        if np.isfinite(v)
    ]
    if len(points) >= 2:
        cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.line(canvas, (x0, y_axis), (x1, y_axis), (0, 0, 0), 3, cv2.LINE_AA)
    for tick_angle, label in ((0.0, "0"), (90.0, "90"), (180.0, "180")):
        tx = int(round(x0 + tick_angle / 180.0 * (x1 - x0)))
        cv2.line(canvas, (tx, y_axis), (tx, y_axis - 12), (0, 0, 0), 2, cv2.LINE_AA)
        _put_label(canvas, label, (max(0, tx - 18), height - 4), scale=0.9, thickness=2)

    markers = [
        ("L", _normalize_angle_marker(left_bound_deg), 2),
        ("R", _normalize_angle_marker(right_bound_deg), 2),
        ("P", _normalize_angle_marker(peak_angle_deg), 3),
    ]
    for label, angle, thickness in markers:
        if angle is None:
            continue
        mx = int(round(x0 + angle / 180.0 * (x1 - x0)))
        cv2.line(canvas, (mx, 0), (mx, y_axis), (0, 0, 0), thickness, cv2.LINE_AA)
        if label == "P":
            idx = int(np.argmin(np.abs(theta_deg - angle)))
            py = points[idx][1] if 0 <= idx < len(points) else y_top + 20
            _put_label(canvas, f"{angle:.1f}", (min(width - 92, mx + 6), max(22, py - 8)), scale=0.8, thickness=2)
        else:
            _put_label(canvas, f"{label}={angle:.1f}", (min(width - 100, mx + 5), 24), scale=0.55, thickness=1)

    return _save_img(save_name, canvas)


def save_sti_angle_overlay(
    sti_u8: np.ndarray,
    line_angle_deg: float,
    fft_peak_angle_deg: float,
    slope: Optional[float],
    peak_value: float,
    save_name: str = "sti_fft_overlay.png",
) -> str:
    h, w = sti_u8.shape[:2]
    vis = cv2.cvtColor(sti_u8, cv2.COLOR_GRAY2BGR)
    cx, cy = w / 2.0, h / 2.0
    length = float(np.hypot(h, w))
    rad = math.radians(line_angle_deg)
    ux, uy = math.cos(rad), math.sin(rad)
    x1 = int(round(cx - length * ux))
    y1 = int(round(cy - length * uy))
    x2 = int(round(cx + length * ux))
    y2 = int(round(cy + length * uy))
    cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)

    text = (
        f"fft_peak={fft_peak_angle_deg:.1f}deg, line={line_angle_deg:.1f}deg, "
        f"slope={('None' if slope is None else f'{slope:.4f}')}, peak={peak_value:.4f}"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = max(5, int(round(0.02 * min(h, w))))
    max_width = max(10, w - 2 * margin)
    font_scale = max(0.35, min(h, w) / 600.0)
    thickness = max(1, int(round(font_scale * 2)))
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    if text_w > max_width:
        font_scale = max(0.2, font_scale * (max_width / float(text_w)))
        thickness = max(1, int(round(font_scale * 2)))
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    text_org = (margin, min(h - baseline - 1, margin + text_h))
    cv2.rectangle(
        vis,
        (text_org[0] - 4, text_org[1] - text_h - baseline - 4),
        (text_org[0] + text_w + 4, text_org[1] + baseline + 4),
        (0, 0, 0),
        -1,
    )
    cv2.putText(vis, text, text_org, font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    return _save_img(save_name, vis)


def _sti_size_tag(sti_u8: np.ndarray, length_px: int, spatial_sample_step: int) -> str:
    h, w = sti_u8.shape[:2]
    return f"step{int(spatial_sample_step)}_len{int(length_px)}_size{w}x{h}"


def save_best_fft_outputs(best: Dict[str, Any]) -> None:
    sti = best.get("sti_raw")
    if sti is None:
        return
    fft_result = best.get("fft_result") or {}

    _save_img("STI_raw.png", sti)
    if "fft_magnitude" in fft_result:
        _save_img("fft_spectrum.png", fft_result["fft_magnitude"])
    if "theta_deg" in fft_result and "response" in fft_result:
        save_fft_curve(
            fft_result["theta_deg"],
            fft_result["response"],
            float(best["fft_peak_angle_deg"]),
            FFT_LEFT_BOUND_DEG,
            FFT_RIGHT_BOUND_DEG,
            save_name="fft_curve.png",
        )
    save_sti_angle_overlay(
        sti,
        float(best["angle"]),
        float(best["fft_peak_angle_deg"]),
        best.get("slope"),
        float(best["fft_peak_value"]),
        save_name="sti_fft_overlay.png",
    )


def analyze_sti_with_fft(sti: np.ndarray, angle_probe: float, fps: float, spatial_sample_step: int) -> Dict[str, Any]:
    fft_result = compute_fft_angle_response(sti)
    slope = fft_result["slope"]
    velocity_px_per_frame = _correct_velocity_px_per_frame(slope, spatial_sample_step)
    return {
        "angle_probe": float(angle_probe),
        "angle": float(fft_result["angle"]),
        "slope": slope,
        "slope_sti": slope,
        "velocity_px_per_frame_corrected": velocity_px_per_frame,
        "score": float(fft_result["score"]),
        "score_mode": SCORE_MODE,
        "fft_peak_angle_deg": float(fft_result["fft_peak_angle_deg"]),
        "fft_peak_value": float(fft_result["fft_peak_value"]),
        "fft_peak_ratio": float(fft_result["fft_peak_ratio"]),
        "fft_min_radius": float(fft_result["fft_min_radius"]),
        "fft_max_radius": float(fft_result["fft_max_radius"]),
        "fft_left_bound_deg": FFT_LEFT_BOUND_DEG,
        "fft_right_bound_deg": FFT_RIGHT_BOUND_DEG,
        "sti_raw": sti,
        "fft_result": fft_result,
        "fps": fps,
    }


def fft_direction_search_on_frames(
    frames: List[np.ndarray],
    fps: float,
    center: Tuple[int, int],
    length_px: int,
    angle_start: float,
    angle_end: float,
    angle_step: float,
    *,
    spatial_sample_step: int,
    max_frames: int,
    save_all_sti: bool,
    save_debug_images: bool,
    verbose: bool,
) -> Dict[str, Any]:
    spatial_count = spatial_sample_count(length_px, spatial_sample_step)
    best: Optional[Dict[str, Any]] = None
    rows: List[Dict[str, Any]] = []
    angle_times: List[Dict[str, float]] = []
    total_t0 = time.perf_counter()
    n_lines = 0

    for angle_probe in _iter_angles(angle_start, angle_end, angle_step):
        t_angle = time.perf_counter()
        n_lines += 1
        sti = build_sti_from_frames(
            frames,
            center,
            length_px,
            angle_probe,
            spatial_sample_step=spatial_sample_step,
        )
        if sti is None:
            continue

        if save_debug_images and save_all_sti:
            tag = _sti_size_tag(sti, length_px, spatial_sample_step)
            _save_img(f"sti_raw/STI_{tag}_a{angle_probe:+06.1f}.png", sti)

        result = analyze_sti_with_fft(sti, angle_probe, fps, spatial_sample_step)
        velocity_mps = None
        row = {
            "probe_angle_deg": float(angle_probe),
            "spatial_sample_step": int(spatial_sample_step),
            "length_px": int(length_px),
            "spatial_sample_count": int(spatial_count),
            "max_frames": int(max_frames),
            "fft_peak_angle_deg": result["fft_peak_angle_deg"],
            "line_angle_deg": result["angle"],
            "slope_sti": result["slope_sti"],
            "velocity_px_per_frame_corrected": result["velocity_px_per_frame_corrected"],
            "velocity_mps": velocity_mps,
            "fft_peak_value": result["fft_peak_value"],
            "fft_peak_ratio": result["fft_peak_ratio"],
            "score": result["score"],
            "score_mode": result["score_mode"],
            "fft_left_bound_deg": result["fft_left_bound_deg"],
            "fft_right_bound_deg": result["fft_right_bound_deg"],
            "fft_min_radius": result["fft_min_radius"],
            "fft_max_radius": result["fft_max_radius"],
        }
        rows.append(row)

        if verbose:
            print(
                f"[角度] a={angle_probe:+06.1f} deg | "
                f"fft_peak={result['fft_peak_angle_deg']:.1f} deg | "
                f"line={result['angle']:.1f} deg | "
                f"peak={result['fft_peak_value']:.4f} | ratio={result['fft_peak_ratio']:.6f}"
            )

        if best is None or float(result["score"]) > float(best["score"]):
            best = result

        angle_times.append({"angle": float(angle_probe), "seconds": float(time.perf_counter() - t_angle)})

    if best is None:
        raise RuntimeError("没有得到有效的 FFT 角度结果")

    best["angle_scores"] = rows
    best["angle_times"] = angle_times
    best["num_lines"] = n_lines
    best["total_time_sec"] = float(time.perf_counter() - total_t0)
    best["spatial_sample_step"] = int(spatial_sample_step)
    best["length_px"] = int(length_px)
    best["spatial_sample_count"] = int(spatial_count)
    best["max_frames"] = int(max_frames)

    csv_path = os.path.join(DEBUG_RUN_DIR or ".", "angle_scores.csv")
    _write_rows_csv(csv_path, rows)
    if verbose:
        print(f"[angles.csv] 已保存每角结果: {csv_path}")

    if save_debug_images:
        save_best_fft_outputs(best)

    return best


def _write_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    _ensure_dir(os.path.dirname(path))
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_flow_overlay(
    frame_bgr: np.ndarray,
    outdir: str,
    center: Tuple[int, int],
    best_angle_deg: float,
    length_px: int,
    slope_px_per_frame: Optional[float],
    spatial_sample_step: int,
    spatial_sample_count_value: int,
    m_per_px: Optional[float],
    fps: Optional[float],
    calib_xyxy: Optional[Tuple[int, int, int, int]] = None,
    calib_real_m: Optional[float] = None,
    center_display: Optional[Tuple[int, int]] = None,
    filename: str = "frame_overlay.png",
    preview_max_side: int = 1280,
) -> None:
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]

    if calib_xyxy and calib_real_m:
        x1, y1, x2, y2 = calib_xyxy
        cv2.line(frame, (x1, y1), (x2, y2), (0, 165, 255), 3, cv2.LINE_AA)
        midx, midy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(
            frame,
            f"Calib {calib_real_m:.2f} m",
            (midx + 10, midy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    (x1, y1, x2, y2), (dx, dy) = _line_endpoints(center, length_px, best_angle_deg)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(
        frame,
        "Velocity Cross-section",
        (min(x1, x2) + 10, min(y1, y2) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    sign = 1 if (slope_px_per_frame is None or slope_px_per_frame >= 0) else -1
    arrow_len = max(60, int(round(length_px * 0.15)))
    start = (int(center[0]), int(center[1]))
    end = (int(center[0] + sign * dx * arrow_len), int(center[1] + sign * dy * arrow_len))
    cv2.arrowedLine(frame, start, end, (0, 255, 0), 4, tipLength=0.1)

    def put(line: str, row: int) -> None:
        y = 35 + row * 30
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    shown_center = center if center_display is None else center_display
    velocity_px_per_frame = _correct_velocity_px_per_frame(slope_px_per_frame, spatial_sample_step)
    v_mps = _velocity_mps(velocity_px_per_frame, m_per_px, fps)

    put(f"center={shown_center}, angle={best_angle_deg:.1f} deg, length={length_px}px", 0)
    put(f"step={spatial_sample_step}, samples={spatial_sample_count_value}", 1)
    put(f"slope_sti={('None' if slope_px_per_frame is None else f'{slope_px_per_frame:.6f}')} sample/frame", 2)
    put(f"v_px/frame={('None' if velocity_px_per_frame is None else f'{velocity_px_per_frame:.6f}')}", 3)
    put(f"m/px={('None' if m_per_px is None else f'{m_per_px:.6f}')}, FPS={('None' if fps is None else f'{fps:.3f}')}", 4)
    put(f"v={abs(v_mps):.4f} m/s" if v_mps is not None else "v=N/A", 5)

    _ensure_dir(outdir)
    out_path = os.path.join(outdir, filename)
    cv2.imwrite(out_path, frame)
    print(f"[overlay] {os.path.abspath(out_path)}")

    max_side = max(h, w)
    if max_side > preview_max_side:
        scale = preview_max_side / float(max_side)
        preview = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        prev_path = os.path.join(outdir, os.path.splitext(filename)[0] + "_preview.png")
        cv2.imwrite(prev_path, preview)
        print(f"[overlay] {os.path.abspath(prev_path)} (preview)")


def _calculate_extended_line(
    center: Tuple[int, int],
    bank_point: Tuple[int, int],
    interval_px: int,
    frame_shape: Tuple[int, int],
) -> List[Tuple[int, int]]:
    if interval_px <= 0:
        raise ValueError("PROBE_INTERVAL_PX 必须为正数")

    h, w = frame_shape
    cx, cy = center
    bx, by = bank_point
    dx = bx - cx
    dy = by - cy
    half_length = math.hypot(dx, dy)
    if half_length < 1:
        raise ValueError("CENTER 与 BANK_POINT 太近，无法生成多点测线")

    ux, uy = dx / half_length, dy / half_length
    points: List[Tuple[int, int]] = []

    for sign in (1, -1):
        dist = 0.0
        while dist <= half_length + 1e-9:
            px = cx + sign * ux * dist
            py = cy + sign * uy * dist
            if not (0 <= px < w and 0 <= py < h):
                break
            pt = (int(round(px)), int(round(py)))
            if pt not in points:
                points.append(pt)
            dist += interval_px

    return points


def batch_probe_along_line_fft(
    frames: List[np.ndarray],
    fps: float,
    center: Tuple[int, int],
    bank_point: Tuple[int, int],
    interval_px: int,
    length_px: int,
    angle_range: Tuple[float, float, float],
    max_frames: int,
    m_per_px: Optional[float],
    fps_override: Optional[float],
    *,
    coord_offset: Tuple[int, int] = (0, 0),
    spatial_sample_step: int = 1,
    save_debug_images: bool = True,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    effective_fps = float(fps_override) if fps_override is not None else float(fps)
    angle_start, angle_end, angle_step = angle_range
    frame_shape = frames[0].shape[:2]
    probe_points_raw = _calculate_extended_line(center, bank_point, interval_px, frame_shape)

    # 以岸边点为首位，其余按与岸边点距离排序，保持和 run.py 接近的输出顺序。
    probe_points = sorted(
        probe_points_raw,
        key=lambda pt: (math.hypot(pt[0] - bank_point[0], pt[1] - bank_point[1]), pt[0], pt[1]),
    )

    results: List[Dict[str, Any]] = []
    ox, oy = int(coord_offset[0]), int(coord_offset[1])

    for idx, point in enumerate(probe_points):
        point_global = (int(point[0] + ox), int(point[1] + oy))
        suffix = f"point_{idx:02d}_x{point_global[0]}_y{point_global[1]}"
        with push_debug_dir(suffix):
            best = fft_direction_search_on_frames(
                frames,
                effective_fps,
                point,
                length_px,
                angle_start,
                angle_end,
                angle_step,
                spatial_sample_step=spatial_sample_step,
                max_frames=max_frames,
                save_all_sti=False,
                save_debug_images=save_debug_images,
                verbose=verbose,
            )

        velocity_px_per_frame = best.get("velocity_px_per_frame_corrected")
        velocity_mps = _velocity_mps(
            velocity_px_per_frame if isinstance(velocity_px_per_frame, (int, float)) else None,
            m_per_px,
            effective_fps,
        )
        speed_m_per_s = abs(velocity_mps) if velocity_mps is not None else None

        row = {
            "index": idx,
            "point_x": point_global[0],
            "point_y": point_global[1],
            "angle_probe_deg": best.get("angle_probe"),
            "line_angle_deg": best.get("angle"),
            "fft_peak_angle_deg": best.get("fft_peak_angle_deg"),
            "spatial_sample_step": int(spatial_sample_step),
            "length_px": int(length_px),
            "spatial_sample_count": int(spatial_sample_count(length_px, spatial_sample_step)),
            "max_frames": int(max_frames),
            "slope_sti": best.get("slope_sti"),
            "velocity_px_per_frame_corrected": velocity_px_per_frame,
            "velocity_mps": velocity_mps,
            "speed_m_per_s": speed_m_per_s,
            "fps": effective_fps,
            "meter_per_pixel": m_per_px,
            "score": best.get("score"),
            "score_mode": best.get("score_mode"),
            "fft_peak_value": best.get("fft_peak_value"),
            "fft_peak_ratio": best.get("fft_peak_ratio"),
            "fft_left_bound_deg": best.get("fft_left_bound_deg"),
            "fft_right_bound_deg": best.get("fft_right_bound_deg"),
        }
        results.append(row)

    csv_path = os.path.join(DEBUG_RUN_DIR or ".", "batch_probe_results.csv")
    _write_rows_csv(csv_path, results)
    print(f"[batch] 结果已保存：{os.path.abspath(csv_path)}")
    return results


def save_batch_overlays(
    frame_bgr: np.ndarray,
    outdir: str,
    center: Tuple[int, int],
    bank_point: Tuple[int, int],
    batch_results: List[Dict[str, Any]],
    *,
    m_per_px: Optional[float],
    default_fps: Optional[float],
) -> None:
    if not batch_results:
        return

    overview = frame_bgr.copy()
    frame_h, frame_w = overview.shape[:2]
    cx, cy = center
    bx, by = bank_point
    another_bank_point = (int(round(2 * cx - bx)), int(round(2 * cy - by)))
    ok, clipped_start, clipped_end = cv2.clipLine((0, 0, frame_w, frame_h), bank_point, another_bank_point)
    if ok:
        cv2.line(overview, clipped_start, clipped_end, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(overview, center, 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(overview, bank_point, 6, (0, 0, 255), -1, cv2.LINE_AA)

    speed_values: List[float] = []
    for row in batch_results:
        speed = row.get("speed_m_per_s")
        if speed is None:
            velocity_mps = _velocity_mps(
                row.get("velocity_px_per_frame_corrected"),
                m_per_px,
                row.get("fps") or default_fps,
            )
            speed = abs(velocity_mps) if velocity_mps is not None else None
        if speed is not None and not _is_speed_out_of_range(float(speed)):
            speed_values.append(abs(float(speed)))

    max_speed = max(speed_values) if speed_values else None
    colors = [(0, 255, 255), (0, 165, 255), (0, 255, 0), (255, 0, 255), (255, 0, 0), (255, 255, 0)]
    overlay_dir = os.path.join(outdir, "batch_overlays")
    _ensure_dir(overlay_dir)

    for row in batch_results:
        angle = row.get("angle_probe_deg")
        if angle is None:
            continue
        idx = int(row.get("index", 0))
        point = (int(row["point_x"]), int(row["point_y"]))
        color = colors[idx % len(colors)]
        length = int(row.get("length_px", LENGTH_PX))
        (x1, y1, x2, y2), (dx, dy) = _line_endpoints(point, length, float(angle))
        cv2.line(overview, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.circle(overview, point, 5, color, -1, cv2.LINE_AA)

        speed = row.get("speed_m_per_s")
        text = f"#{idx} " + ("N/A" if speed is None else f"{float(speed):.2f} m/s")
        cv2.putText(overview, text, (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        min_arrow_len = max(20, int(round(length * 0.2)))
        max_arrow_len = max(min_arrow_len + 1, int(round(length * 0.7)))
        arrow_len = min_arrow_len
        if speed is not None and max_speed and max_speed > 0:
            scale = abs(float(speed)) / max_speed
            arrow_len = int(round(min_arrow_len + scale * (max_arrow_len - min_arrow_len)))

        slope = row.get("slope_sti")
        sign = 1 if (slope is None or float(slope) >= 0) else -1
        start = (int(point[0]), int(point[1]))
        end = (int(point[0] + sign * dx * arrow_len), int(point[1] + sign * dy * arrow_len))
        cv2.arrowedLine(overview, start, end, color, 3, tipLength=0.2)

    overview_path = os.path.join(overlay_dir, "batch_overview.png")
    cv2.imwrite(overview_path, overview)
    print(f"[batch overlay] 总览图已保存：{os.path.abspath(overview_path)}")


def save_single_summary(
    outdir: str,
    best: Dict[str, Any],
    *,
    length_px: int,
    spatial_sample_step: int,
    spatial_sample_count_value: int,
    max_frames: int,
    m_per_px: Optional[float],
) -> None:
    slope_sti = best.get("slope_sti", best.get("slope"))
    velocity_px_per_frame = best.get("velocity_px_per_frame_corrected")
    if velocity_px_per_frame is None:
        velocity_px_per_frame = _correct_velocity_px_per_frame(
            slope_sti if isinstance(slope_sti, (int, float)) else None,
            spatial_sample_step,
        )
    fps = best.get("fps")
    velocity_mps = _velocity_mps(
        velocity_px_per_frame if isinstance(velocity_px_per_frame, (int, float)) else None,
        m_per_px,
        fps if isinstance(fps, (int, float)) else None,
    )

    row = {
        "spatial_sample_step": spatial_sample_step,
        "length_px": length_px,
        "spatial_sample_count": spatial_sample_count_value,
        "max_frames": max_frames,
        "slope_sti": slope_sti,
        "velocity_px_per_frame_corrected": velocity_px_per_frame,
        "velocity_mps": velocity_mps,
        "fps": fps,
        "meter_per_pixel": m_per_px,
        "angle_probe_deg": best.get("angle_probe"),
        "line_angle_deg": best.get("angle"),
        "fft_peak_angle_deg": best.get("fft_peak_angle_deg"),
        "score": best.get("score"),
        "score_mode": best.get("score_mode"),
        "fft_peak_value": best.get("fft_peak_value"),
        "fft_peak_ratio": best.get("fft_peak_ratio"),
        "fft_left_bound_deg": best.get("fft_left_bound_deg"),
        "fft_right_bound_deg": best.get("fft_right_bound_deg"),
        "fft_min_radius": best.get("fft_min_radius"),
        "fft_max_radius": best.get("fft_max_radius"),
    }
    summary_path = os.path.join(outdir, "summary.csv")
    _write_rows_csv(summary_path, [row])
    print(f"[summary] {os.path.abspath(summary_path)}")


def main() -> None:
    if not os.path.isfile(VIDEO):
        raise FileNotFoundError(f"视频不存在: {VIDEO}")
    if START_FRAME is not None and START_TIME_SEC is not None:
        raise ValueError("START_FRAME 与 START_TIME_SEC 只能设置一个")

    outdir = init_debug_dir(tag="stiv-fft")
    frame0_full = _read_first_frame(VIDEO)
    frame_h, frame_w = frame0_full.shape[:2]

    roi_box: Optional[Tuple[int, int, int, int]] = None
    roi_offset = (0, 0)
    center_proc = CENTER
    bank_point_proc = BANK_POINT
    calib_line_proc = CALIB_LINE_XYXY

    if ROI is not None:
        roi_box = _normalize_roi(ROI, frame_w, frame_h)
        if not _point_in_roi(CENTER, roi_box):
            raise ValueError(f"CENTER 点落在 ROI 外: CENTER={CENTER}, ROI={roi_box}")
        center_proc = _global_to_local_point(CENTER, roi_box)
        bank_point_proc = _global_to_local_point(BANK_POINT, roi_box)
        calib_line_proc = _global_to_local_line(CALIB_LINE_XYXY, roi_box)
        roi_offset = (roi_box[0], roi_box[1])

    first_frame_proc = _read_first_frame(VIDEO, roi_box)
    check_w = frame_w if roi_box is None else roi_box[2] - roi_box[0]
    check_h = frame_h if roi_box is None else roi_box[3] - roi_box[1]
    spatial_count = spatial_sample_count(LENGTH_PX, SPATIAL_SAMPLE_STEP)

    for angle in _iter_angles(ANGLE_START, ANGLE_END, ANGLE_STEP):
        if not _line_fully_inside_frame(center_proc, LENGTH_PX, angle, check_w, check_h):
            raise ValueError(
                f"CENTER 对应测速线超出ROI: center={CENTER}, angle={angle}, "
                f"LENGTH_PX={LENGTH_PX}, ROI={roi_box}"
            )

    print(f"[out] 所有步骤图将保存到：{outdir}")
    print(
        f"[cfg] CENTER={CENTER}, LENGTH_PX={LENGTH_PX}, SPATIAL_SAMPLE_STEP={SPATIAL_SAMPLE_STEP}, "
        f"spatial_sample_count={spatial_count}, ANGLES=({ANGLE_START},{ANGLE_END},{ANGLE_STEP}), "
        f"MAX_FRAMES={MAX_FRAMES}, USE_ROI={USE_ROI}"
    )
    print(f"[cfg] ROI={ROI}")
    print(f"[cfg] START_FRAME={START_FRAME}, START_TIME_SEC={START_TIME_SEC}")
    print(
        f"[cfg] FFT_LEFT_BOUND_DEG={FFT_LEFT_BOUND_DEG}, "
        f"FFT_RIGHT_BOUND_DEG={FFT_RIGHT_BOUND_DEG}, "
        f"FFT_PEAK_SEARCH_RANGE_DEG={FFT_PEAK_SEARCH_RANGE_DEG}"
    )

    m_per_px = SCALE_M_PER_PIXEL
    if m_per_px is None and (CALIB_REAL_M is not None and CALIB_LINE_XYXY is not None):
        m_per_px = compute_scale_from_first_frame(VIDEO, CALIB_LINE_XYXY, CALIB_REAL_M)
    if m_per_px is not None:
        print(f"[scale] 使用 SCALE_M_PER_PIXEL={m_per_px:.6f} m/px")
    else:
        print("[scale] 未提供比例尺；将仅输出像素单位的斜率，不计算 m/s")

    frames, video_fps = _load_video_frames(
        VIDEO,
        MAX_FRAMES,
        start_frame=START_FRAME,
        start_time_sec=START_TIME_SEC,
        roi=roi_box,
    )
    effective_fps = float(FPS) if FPS is not None else float(video_fps)
    print(f"[fps] 使用 FPS={effective_fps:.6f}")

    if USE_BATCH_LINE_PROBING:
        results = batch_probe_along_line_fft(
            frames,
            effective_fps,
            center_proc,
            bank_point_proc,
            PROBE_INTERVAL_PX,
            LENGTH_PX,
            (ANGLE_START, ANGLE_END, ANGLE_STEP),
            MAX_FRAMES,
            m_per_px,
            FPS,
            coord_offset=roi_offset,
            spatial_sample_step=SPATIAL_SAMPLE_STEP,
            save_debug_images=SAVE_DEBUG_IMAGES,
            verbose=VERBOSE,
        )

        print("\n====== 多点测速结果 ======")
        for row in results:
            speed_txt = "N/A" if row["speed_m_per_s"] is None else f"{row['speed_m_per_s']:.4f} m/s"
            print(
                f"#{row['index']:02d} pt=({row['point_x']},{row['point_y']}) "
                f"len={row['length_px']}px step={row['spatial_sample_step']} "
                f"samples={row['spatial_sample_count']} angle={row['angle_probe_deg']}° "
                f"fft_peak={row['fft_peak_angle_deg']}° line={row['line_angle_deg']}° "
                f"slope_sti={row['slope_sti']} sample/frame "
                f"v_px/frame={row['velocity_px_per_frame_corrected']} speed={speed_txt} score={row['score']}"
            )

        overlay_results = results
        if roi_box is not None:
            ox, oy = roi_offset
            overlay_results = []
            for row in results:
                local_row = dict(row)
                local_row["point_x"] = int(local_row["point_x"]) - ox
                local_row["point_y"] = int(local_row["point_y"]) - oy
                overlay_results.append(local_row)

        save_batch_overlays(
            first_frame_proc,
            outdir,
            center_proc,
            bank_point_proc,
            overlay_results,
            m_per_px=m_per_px,
            default_fps=effective_fps,
        )
        return

    best = fft_direction_search_on_frames(
        frames,
        effective_fps,
        center_proc,
        LENGTH_PX,
        ANGLE_START,
        ANGLE_END,
        ANGLE_STEP,
        spatial_sample_step=SPATIAL_SAMPLE_STEP,
        max_frames=MAX_FRAMES,
        save_all_sti=SAVE_ALL_STI,
        save_debug_images=SAVE_DEBUG_IMAGES,
        verbose=VERBOSE,
    )

    save_flow_overlay(
        first_frame_proc,
        outdir,
        center_proc,
        float(best.get("angle_probe", best["angle"])),
        LENGTH_PX,
        best.get("slope"),
        SPATIAL_SAMPLE_STEP,
        spatial_count,
        m_per_px,
        effective_fps,
        calib_xyxy=calib_line_proc,
        calib_real_m=CALIB_REAL_M,
        center_display=CENTER,
        filename="frame_overlay.png",
        preview_max_side=1280,
    )

    print("\n====== 最终结果 ======")
    print(f"中心点: {CENTER}")
    print(f"LENGTH_PX: {LENGTH_PX} px")
    print(f"SPATIAL_SAMPLE_STEP: {SPATIAL_SAMPLE_STEP}")
    print(f"spatial_sample_count: {spatial_count}")
    print(f"MAX_FRAMES: {MAX_FRAMES}")
    print(f"测速线方向: {best.get('angle_probe')} °")
    print(f"FFT 峰值角度: {best.get('fft_peak_angle_deg')} °")
    print(f"STI 结果纹理角度: {best.get('angle')} °")
    print(f"FFT 峰值强度: {best.get('fft_peak_value'):.6f}")
    print(f"FFT 主峰占比: {best.get('fft_peak_ratio'):.6f}")
    print(f"用于筛选的得分: {best.get('score'):.6f} ({best.get('score_mode')})")

    slope_sti = best.get("slope_sti", best.get("slope"))
    velocity_px_per_frame = best.get("velocity_px_per_frame_corrected")
    print(f"STI 斜率 slope_sti (sample/frame): {slope_sti if slope_sti is not None else 'None'}")
    print(
        "校正后像素速度 velocity_px_per_frame_corrected: "
        f"{velocity_px_per_frame if velocity_px_per_frame is not None else 'None'} px/frame"
    )
    if velocity_px_per_frame is not None and effective_fps:
        print(
            f"像素速度: {velocity_px_per_frame * effective_fps:.4f} px/s "
            f"(velocity_px_per_frame={velocity_px_per_frame:.6f}, FPS={effective_fps:.3f})"
        )
    else:
        print("未计算像素速度：缺少 slope_sti 或 FPS。")

    v_mps = _velocity_mps(
        velocity_px_per_frame if isinstance(velocity_px_per_frame, (int, float)) else None,
        m_per_px,
        effective_fps,
    )
    if v_mps is not None:
        print(
            f"速度估计: {v_mps:.4f} m/s "
            f"(velocity_px_per_frame={velocity_px_per_frame:.6f}, m/px={m_per_px:.6f}, FPS={effective_fps:.3f})"
        )
    else:
        print("未计算速度：缺少 velocity_px_per_frame_corrected 或 m/px 或 FPS。")

    save_single_summary(
        outdir,
        best,
        length_px=LENGTH_PX,
        spatial_sample_step=SPATIAL_SAMPLE_STEP,
        spatial_sample_count_value=spatial_count,
        max_frames=MAX_FRAMES,
        m_per_px=m_per_px,
    )

    times = best.get("angle_times") or []
    print(f"测速线数量: {best.get('num_lines', 0)}")
    print(f"总用时: {best.get('total_time_sec', 0.0):.3f} s")
    if times:
        avg = sum(t["seconds"] for t in times) / len(times)
        slow = max(times, key=lambda t: t["seconds"])
        print(f"单条平均用时: {avg:.3f} s，最慢: {slow['angle']:.1f}° -> {slow['seconds']:.3f} s")
    print("所有步骤图已写入输出目录。")


if __name__ == "__main__":
    main()
    t1 = time.perf_counter()
    print(f"[TIME] total = {t1 - t0:.3f} s")
