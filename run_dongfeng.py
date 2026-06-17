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

from stiv_adapt.core import _save_img, init_debug_dir, push_debug_dir, spatial_sample_count

try:
    import pandas as _pandas_probe  # noqa: F401
except ModuleNotFoundError:
    import sys

    class _MissingPandasModule:
        def DataFrame(self, *args: Any, **kwargs: Any) -> Any:
            raise ModuleNotFoundError("No module named 'pandas'")

    sys.modules["pandas"] = _MissingPandasModule()

from stiv_adapt.search import _adaptive_direction_search_on_frames, _load_video_frames

t0 = time.perf_counter()

# ========== User config ==========
VIDEO = r"D:\Desktop\东风渠\测流视频\20260529\110000_undistort.mp4"
CONFIG_PATH = "Config.ini"

# Windows 中文控制台通常使用 GBK。若你的终端是 UTF-8，可改成 "utf-8"。
OUTPUT_ENCODING: Optional[str] = "utf-8"

# Dongfeng layout: fixed visible water-surface station range.
DONGFENG_DIST_START_M = 3.1
DONGFENG_DIST_END_M = 20.6
DONGFENG_DIST_STEP_M = 0.5

# STI line params. LENGTH_PX is still the source-image measuring line length.
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

VOTE_THETA_RES_DEG = 0.05
VOTE_K_RATIO: float = 0.55
VOTE_THETA_RANGE = (80, 87)


#Hough 法线角 theta 的粗搜到精搜
USE_COARSE_TO_FINE_THETA: bool = True
VOTE_THETA_COARSE_RES_DEG: float = 0.5  #粗搜步长
VOTE_THETA_FINE_RES_DEG: float = 0.01    #精搜步长
VOTE_THETA_FINE_HALF_WIDTH_DEG: float = 1.0   #精搜半宽
# =================================


def configure_output_encoding() -> None:
    """Make Chinese progress logs match the current Windows console encoding."""
    if not OUTPUT_ENCODING:
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding=OUTPUT_ENCODING, errors="replace")
        except Exception:
            pass


def load_config_ini(path: str) -> Dict[str, Dict[str, str]]:
    """Read the simple key=value Config.ini used by Hydroview."""
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
    """Minimal water-plane geometry for Dongfeng testing-line placement."""

    def __init__(self, cfg: Dict[str, Dict[str, str]], frame_w: int, frame_h: int):
        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)
        self.fx = cfg_float(cfg, "MeasuringSettings", "fx")
        self.fy = cfg_float(cfg, "MeasuringSettings", "fy")
        self.cx = cfg_float(cfg, "MeasuringSettings", "cx")
        self.cy = cfg_float(cfg, "MeasuringSettings", "cy")
        self.img_w = cfg_int(cfg, "MeasuringSettings", "ImgWidth", frame_w)
        self.img_h = cfg_int(cfg, "MeasuringSettings", "ImgHeight", frame_h)
        self.cam_position_to_ref = cfg_int(cfg, "MeasuringSettings", "CamPositionToRef")
        if self.cam_position_to_ref != 1:
            raise ValueError("run_dongfeng.py currently expects CamPositionToRef=1")

        stage_m = cfg_float(cfg, "UserSettings", "StageInput")
        camera_height_m = cfg_float(cfg, "MeasuringSettings", "CameraHeight")
        self.height_above_water_m = camera_height_m - stage_m
        if self.height_above_water_m <= 0:
            raise ValueError(
                f"invalid camera height above water: CameraHeight={camera_height_m}, StageInput={stage_m}"
            )

        self.camera_dist_m = cfg_float(cfg, "MeasuringSettings", "CameraDist")
        pitch_deg = cfg_float(cfg, "MeasuringSettings", "TiltSensorPitch")
        pitch_deg += cfg_float(cfg, "MeasuringSettings", "dPitch", 0.0)
        self.pitch_rad = math.radians(pitch_deg)

    def station_to_pixel_y(self, station_m: float) -> float:
        """Convert cross-section station distance to image y at the TL center x."""
        forward_m = float(station_m) - self.camera_dist_m
        if forward_m <= 0:
            raise ValueError(f"station {station_m:.3f} m is behind the camera reference")
        beta = math.atan(self.height_above_water_m / forward_m) - self.pitch_rad
        return self.cy + self.fy * math.tan(beta)

    def ground_point(self, u: float, v: float) -> Tuple[float, float]:
        """Return water-plane point as (flow_axis_m, station_m)."""
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
    """Return OpenCV tipLength ratio for a fixed-size arrow head."""
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


def run_adaptive_search_quietly(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Run the shared search code while hiding non-ASCII library progress logs."""
    if not SUPPRESS_LIBRARY_OUTPUT:
        return _adaptive_direction_search_on_frames(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return _adaptive_direction_search_on_frames(*args, **kwargs)


def build_dongfeng_testing_lines(
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


def save_dongfeng_overlay(video_path: str, outdir: str, results: List[Dict[str, Any]], length_px: int) -> None:
    if not results:
        return
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("[叠加图] 无法读取首帧")
        return

    colors = [
        (0, 255, 255),
        (0, 165, 255),
        (0, 255, 0),
        (255, 0, 255),
        (255, 0, 0),
        (255, 255, 0),
    ]
    valid_speeds = [
        abs(float(row["speed_m_per_s"]))
        for row in results
        if row.get("speed_m_per_s") is not None and not speed_out_of_range(float(row["speed_m_per_s"]))
    ]
    max_speed = max(valid_speeds) if valid_speeds else None

    for row in results:
        speed = row.get("speed_m_per_s")
        out_of_range = speed_out_of_range(float(speed)) if speed is not None else False
        point = (int(row["point_x"]), int(row["point_y"]))
        angle = float(row.get("angle_probe_deg") or ANGLE_START)
        color = colors[int(row["index"]) % len(colors)]

        if out_of_range:
            cross_size = max(6, int(round(length_px * 0.08)))
            cv2.line(frame, (point[0] - cross_size, point[1] - cross_size), (point[0] + cross_size, point[1] + cross_size), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (point[0] - cross_size, point[1] + cross_size), (point[0] + cross_size, point[1] - cross_size), (0, 0, 255), 2, cv2.LINE_AA)
            continue

        (x1, y1, x2, y2), direction = line_endpoints(point, length_px, angle)
        cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)
        label = f"{row['station_m']:.1f}m"
        if speed is not None:
            label += f" {float(speed):.2f}m/s"
        cv2.putText(frame, label, (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        slope = row.get("slope_sti")
        sign = 1 if slope is None or float(slope) >= 0 else -1
        min_arrow_len = max(20, int(round(length_px * 0.08)))
        max_arrow_len = max(min_arrow_len + 1, int(round(length_px * 0.25)))
        arrow_len = min_arrow_len
        if speed is not None and max_speed and max_speed > 0:
            arrow_len = int(round(min_arrow_len + abs(float(speed)) / max_speed * (max_arrow_len - min_arrow_len)))
        dx, dy = direction
        end = (int(point[0] + sign * dx * arrow_len), int(point[1] + sign * dy * arrow_len))
        cv2.arrowedLine(frame, point, end, color, 2, tipLength=fixed_arrow_tip_ratio(arrow_len))




    overlay_dir = os.path.join(outdir, "dongfeng_overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    out_path = os.path.join(overlay_dir, "dongfeng_world_layout_overview.png")
    cv2.imwrite(out_path, frame)
    print(f"[叠加图] {os.path.abspath(out_path)}")


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

    testing_lines_global = build_dongfeng_testing_lines(cfg, frame_w, frame_h, LENGTH_PX)
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
    print(
        f"[配置] 东风渠现实等距布线: 起点距={DONGFENG_DIST_START_M:.1f}..{DONGFENG_DIST_END_M:.1f} m, "
        f"间隔={DONGFENG_DIST_STEP_M:.1f} m, 数量={len(testing_lines_global)}"
    )
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
        with push_debug_dir(suffix):
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
                save_all_sti=SAVE_ALL_STI,
                top_k_candidates=TOP_K_CANDIDATES,
                score_mode=SCORE_MODE,
                save_debug_images=SAVE_DEBUG_IMAGES,
            )
            raw_sti = best.get("sti_raw")
            if raw_sti is not None:
                h, w = raw_sti.shape[:2]
                _save_img(
                    f"{suffix}_STI_step{SPATIAL_SAMPLE_STEP}_len{LENGTH_PX}_size{w}x{h}.png",
                    raw_sti,
                )

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
    save_dongfeng_overlay(process_video, outdir, overlay_rows, LENGTH_PX)

    print("[完成] 东风渠现实等距测速线处理完成。")


if __name__ == "__main__":
    main()
    t1 = time.perf_counter()
    print(f"[耗时] total = {t1 - t0:.3f} s")
