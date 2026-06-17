# -*- coding: utf-8 -*-
import csv
import os
import math
import time
import cv2
from typing import Optional, Tuple, List, Dict
from stiv_adapt.search import adaptive_direction_search
from stiv_adapt.core import init_debug_dir, spatial_sample_count
t0 = time.perf_counter()
# ========== 用户配置区（按需修改） ==========
VIDEO = r"D:\Desktop\东风渠\测流视频\20260529\110000_undistort.mp4"
CENTER: Tuple[int, int] =(1919, 1049)# ← 手动中心点（像素坐标）
#多点测速参数
USE_BATCH_LINE_PROBING = True # ← 开启多点测速
BANK_POINT: Tuple[int, int] =(1919, 193)#(797, 583) # 岸边点（与 CENTER 组成测速直线）
PROBE_INTERVAL_PX = 100 # 两测点之间的像素间隔（从中心点向两端延伸）

# STI 测线参数（角度搜索范围：线方向）
LENGTH_PX = 1500
ANGLE_START, ANGLE_END, ANGLE_STEP =0, 0, 1   # 遍历的“测速线角度”
MAX_FRAMES = 750
SPATIAL_SAMPLE_STEP = 2
ARROW_HEAD_PX = 10

#从某帧或某秒开始
START_FRAME: Optional[int] = None
START_TIME_SEC: Optional[float] = None

# 最佳测速线方向选择策略：peak_votes / peak_ratio
SCORE_MODE: str = "peak_ratio"
# 边缘提取方式：可选 "canny" 或 "sobel"
EDGE_METHOD: str = "sobel"
# Sobel 得分阈值系数（thr = mu + k_sigma * sigma）
K_SIGMA: float = 2

USE_ROI = True
ROI_RADIUS_FRAC: float = 0.9  # ROI 半径比例（相对 min(H, W)/2），需开启 USE_ROI 才生效
ROI: Optional[Tuple[int, int, int, int]] = None#(1400,200,2500,1200)  # 矩形 ROI: (x0, y0, x1, y1)，None 表示关闭
VERBOSE = True
# 对称性评分开关：True 时使用 E_asym 二次筛选最佳角度
USE_E_ASYM: bool = False
# 单调性评分开关：True 时启用 M_mono 判据（优先于对称性）
USE_M_MONO: bool = False
# M_mono 单调性评分的峰值阈值比例（0~1）：越高越靠近峰顶
M_MONO_PEAK_RATIO: float = 0.5
# 角度候选的得分截取数量（默认取前10名）
TOP_K_CANDIDATES: int = 3


# 测速线方向搜索方法：可选 "hough" 或 "autocorr"
DIRECTION_METHOD: str = "hough"
# 单点测速：保存所有 STI 中间结果（按步骤分文件夹）
SAVE_ALL_STI: bool = True
# 调试图像总开关：False 时不保存角度循环中的中间图（优先级高于 SAVE_ALL_STI）
SAVE_DEBUG_IMAGES: bool = True

# 速度阈值设置（m/s），可按需修改；留 None 表示不限制
V_MIN: Optional[float] = None
V_MAX: Optional[float] = None

# 帧率（建议手动给准值；留 None 则使用视频元数据）
FPS: Optional[float] = None

# 比例尺：二选一
SCALE_M_PER_PIXEL: Optional[float] = None  # A) 直接给（m/px）；不想手填则设 None 走 B)
CALIB_REAL_M: Optional[float] = None     # B) 首帧两点标定（米）
CALIB_LINE_XYXY: Optional[Tuple[int, int, int, int]] = (476, 835,3356, 809)#CRR(445, 1321, 3085, 1444)
#投票霍夫的可调参数（法线角 θ 的设置）——
VOTE_THETA_RES_DEG = 0.1              # 精搜/旧模式角度分辨率（度）
VOTE_K_RATIO: float = 0.55             # 用比例阈值 K=0.55*R
VOTE_THETA_RANGE = (45,135)        # 有效法线角范围 [min, max)
USE_COARSE_TO_FINE_THETA: bool = True
VOTE_THETA_COARSE_RES_DEG: float = 1.0
VOTE_THETA_FINE_RES_DEG: float = 0.1
VOTE_THETA_FINE_HALF_WIDTH_DEG: float = 1.0
# ==========================================
def compute_scale_from_first_frame(video_path: str,
                                   xyxy: Tuple[int, int, int, int],
                                   real_meters: float) -> float:
    """在视频首帧上用两点像素距离和真实距离求 m/px。"""
    cap = cv2.VideoCapture(video_path)
    ok, frame0 = cap.read()
    cap.release()#确保视频可以读取
    if not ok:
        raise RuntimeError("无法读取视频首帧用于标定")
    x1, y1, x2, y2 = xyxy
    px = math.hypot(x2 - x1, y2 - y1)
    if px < 1:
        raise RuntimeError("标定两点太近或坐标不正确")
    m_per_px = real_meters / px
    print(f"[calib] 像素距离={px:.2f}px, 真实距离={real_meters:.3f}m -> SCALE_M_PER_PIXEL={m_per_px:.6f} m/px")
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


def _crop_video_to_roi(video_path: str, out_path: str, roi: Tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = roi
    crop_w, crop_h = x1 - x0, y1 - y0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not fps or math.isnan(fps) or math.isinf(fps):
        fps = 30.0
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (crop_w, crop_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建 ROI 视频: {out_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame[y0:y1, x0:x1])
    finally:
        cap.release()
        writer.release()


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


def _line_fully_inside_frame(center: Tuple[int, int], length_px: int, angle_deg: float, frame_w: int, frame_h: int) -> bool:
    (x1, y1, x2, y2), _ = _line_endpoints(center, length_px, angle_deg)
    return (0 <= x1 < frame_w and 0 <= y1 < frame_h and 0 <= x2 < frame_w and 0 <= y2 < frame_h)

def _is_speed_out_of_range(speed: Optional[float]) -> bool:
    """依据 V_MIN/V_MAX 判断速度是否超出允许范围（按绝对值比较）。"""
    if speed is None:
        return False
    abs_speed = abs(speed)
    if V_MIN is not None and abs_speed < V_MIN:
        return True
    if V_MAX is not None and abs_speed > V_MAX:
        return True
    return False


def _correct_velocity_px_per_frame(slope_sti: Optional[float],
                                   spatial_sample_step: int) -> Optional[float]:
    """把 STI 采样点/frame 斜率换算成原图 px/frame。"""
    if slope_sti is None:
        return None
    return float(slope_sti) * float(spatial_sample_step)


def _velocity_mps(velocity_px_per_frame: Optional[float],
                  m_per_px: Optional[float],
                  fps: Optional[float]) -> Optional[float]:
    """按原图 px/frame 速度换算 m/s，保留方向符号。"""
    if velocity_px_per_frame is None or m_per_px is None or fps is None:
        return None
    return float(velocity_px_per_frame) * float(m_per_px) * float(fps)


def _line_endpoints(center, length_px, angle_deg):
    """"根据中心点和线长度，计算两端点坐标和方向向量"""
    cx, cy = center
    half = length_px / 2.0
    rad  = math.radians(angle_deg) #转弧度
    dx, dy = math.cos(rad), math.sin(rad)#单位方向向量
    x1 = int(round(cx - half*dx)); y1 = int(round(cy - half*dy))
    x2 = int(round(cx + half*dx)); y2 = int(round(cy + half*dy))
    return (x1, y1, x2, y2), (dx, dy)


def _fixed_arrow_tip_ratio(arrow_len_px: float) -> float:
    """Return OpenCV tipLength ratio for a fixed-size arrow head."""
    return min(0.45, ARROW_HEAD_PX / max(float(arrow_len_px), 1.0))


def save_flow_overlay(
    video_path: str,
    outdir: str,
    center: tuple,                    # (cx, cy)
    best_angle_deg: float,            # 你的最佳角度（线方向）
    length_px: int,                   # 测线像素长度
    slope_px_per_frame: float|None,   # dx/dy (STI sample/frame)
    spatial_sample_step: int,         # 测速线方向空间抽样步长
    spatial_sample_count: int,        # 抽样后的 STI 空间宽度
    m_per_px: float|None,             # 比例尺，可为 None
    fps: float|None,                  # 帧率，可为 None
    calib_xyxy: tuple|None=None,      # (x1,y1,x2,y2)
    calib_real_m: float|None=None,    # 真实距离（米）
    center_display: tuple|None=None,  # 显示时使用的中心点（用于保持原图坐标）
    filename: str="frame_overlay.png",
    preview_max_side: int=1280
):
    """读取首帧，叠加测速线"""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("[overlay] 无法读取首帧"); return

    H, W = frame.shape[:2]

    # —— 可选：标定线（橙色） ——
    if calib_xyxy and calib_real_m:
        x1,y1,x2,y2 = calib_xyxy
        cv2.line(frame, (x1,y1), (x2,y2), (0,165,255), 3, cv2.LINE_AA)
        midx, midy = (x1+x2)//2, (y1+y2)//2
        cv2.putText(frame, f"Calib {calib_real_m:.2f} m",
                    (midx+10, midy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0,165,255), 2, cv2.LINE_AA)

    # —— 测速截面（黄色实线） ——
    (x1,y1,x2,y2), (dx,dy) = _line_endpoints(center, length_px, best_angle_deg)
    cv2.line(frame, (x1,y1), (x2,y2), (0,255,255), 4, cv2.LINE_AA)
    cv2.putText(frame, "Velocity Cross-section",
                (min(x1,x2)+10, min(y1,y2)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)

    # —— 流动方向箭头（绿色） ——
    sign = 1 if (slope_px_per_frame is None or slope_px_per_frame >= 0) else -1
    arrow_len = max(60, int(round(length_px * 0.15)))
    #arrow_len = 4
    start = (int(center[0]), int(center[1]))
    end   = (int(center[0] + sign*dx*arrow_len), int(center[1] + sign*dy*arrow_len))
    cv2.arrowedLine(frame, start, end, (0,255,0), 4, tipLength=_fixed_arrow_tip_ratio(arrow_len))

    # —— 左上角信息：slope / m/px / FPS / v(m/s) ——
    def put(line, row):
        """在叠加图左上角写入一行文本。"""
        y = 35 + row*30
        cv2.putText(frame, line, (15,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20,20,20), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (15,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    shown_center = center if center_display is None else center_display
    velocity_px_per_frame = _correct_velocity_px_per_frame(slope_px_per_frame, spatial_sample_step)
    v_mps = _velocity_mps(velocity_px_per_frame, m_per_px, fps)

    put(f"center={shown_center}, angle={best_angle_deg:.1f} deg, length={length_px}px", 0)
    put(f"step={spatial_sample_step}, samples={spatial_sample_count}", 1)
    put(f"slope_sti = {('None' if slope_px_per_frame is None else f'{slope_px_per_frame:.6f}')} sample/frame", 2)
    put(f"v_px/frame = {('None' if velocity_px_per_frame is None else f'{velocity_px_per_frame:.6f}')}", 3)
    put(f"m/px={('None' if m_per_px is None else f'{m_per_px:.6f}')}, FPS={('None' if fps is None else f'{fps:.3f}')}", 4)

    if v_mps is not None:
        put(f"v = {abs(v_mps):.4f} m/s", 5)
    else:
        put("v = N/A (缺少 slope/mpp/FPS)", 5)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, filename)
    cv2.imwrite(out_path, frame)
    print(f"[overlay] {os.path.abspath(out_path)}")

    # 预览缩放
    max_side = max(H, W)
    if max_side > preview_max_side:
        scale = preview_max_side / float(max_side)
        prev = cv2.resize(frame, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
        prev_path = os.path.join(outdir, os.path.splitext(filename)[0] + "_preview.png")
        cv2.imwrite(prev_path, prev)
        print(f"[overlay] {os.path.abspath(prev_path)} (preview)")


def save_batch_overlays(
    video_path: str,
    outdir: str,
    center: Tuple[int, int],
    bank_point: Tuple[int, int],
    batch_results: List[Dict[str, object]],
    *,
    m_per_px: Optional[float],
    default_fps: Optional[float],
) -> None:
    """
    功能：在视频首帧上绘制“多点测速”的整体结果，并为每个测点生成单独的叠加图。
    主要可视化要素：
        - 中心点与岸边点、基准测线；
        - 每个探测点的测速截线、点位标识；
        - 速度文本（m/s）；
        - 按速度大小归一化的方向箭头。
    """

    # 若结果为空则直接返回
    if not batch_results:
        return

    # === 1. 读取视频首帧 ===
    cap = cv2.VideoCapture(video_path)
    ok, frame0 = cap.read()
    cap.release()
    if not ok:
        print("[batch overlay] 无法读取首帧，跳过叠加图保存")
        return

    # 复制首帧作为绘图底图
    overview = frame0.copy()

    # === 2. 绘制中心点、岸边点及对岸对称点的基准连线 ===
    frame_h, frame_w = overview.shape[:2]
    cx, cy = center
    bx, by = bank_point
    # 计算岸边点关于中心点的对称点，形成完整的横跨两岸的基准线
    another_bank_point = (int(round(2 * cx - bx)), int(round(2 * cy - by)))
    # 使用 clipLine 保证绘制端点仍位于画面内
    ok, clipped_start, clipped_end = cv2.clipLine((0, 0, frame_w, frame_h), bank_point, another_bank_point)
    if ok:
        cv2.line(overview, clipped_start, clipped_end, (255, 255, 0), 2, cv2.LINE_AA)   # 青色连线
    cv2.circle(overview, center, 6, (0, 0, 255), -1, cv2.LINE_AA)                       # 中心点红圆
    cv2.circle(overview, bank_point, 6, (0, 0, 255), -1, cv2.LINE_AA)                   # 岸边点红圆
    if 0 <= another_bank_point[0] < frame_w and 0 <= another_bank_point[1] < frame_h:
        cv2.circle(overview, another_bank_point, 6, (0, 0, 255), -1, cv2.LINE_AA)       # 对岸点红圆

    # === 3. 计算速度绝对值，用于后续箭头长度归一化 ===
    speed_values: List[float] = []
    for row in batch_results:
        spd = row.get("speed_m_per_s")

        # 若没有直接给出速度，则用校正后的原图 px/frame 速度计算
        if spd is None:
            slope = row.get("slope_px_per_frame")
            step = int(row.get("spatial_sample_step", SPATIAL_SAMPLE_STEP))
            velocity_px_per_frame = row.get("velocity_px_per_frame_corrected")
            if velocity_px_per_frame is None:
                velocity_px_per_frame = _correct_velocity_px_per_frame(slope, step)
            fps_here = row.get("fps") or default_fps
            velocity_mps = _velocity_mps(velocity_px_per_frame, m_per_px, fps_here)
            if velocity_mps is not None:
                spd = abs(velocity_mps)

        # 若得到有效速度，则保存至临时字段 "_overlay_speed_mps"
        if spd is not None:
            try:
                overlay_speed = float(spd)
            except (TypeError, ValueError):
                continue
            row["_overlay_speed_mps"] = overlay_speed

            if not _is_speed_out_of_range(overlay_speed):
                speed_values.append(abs(overlay_speed))

    max_speed = max(speed_values) if speed_values else None

    # === 4. 定义颜色组，用于不同测点区分 ===
    colors = [
        (0, 255, 255),
        (0, 165, 255),
        (0, 255, 0),
        (255, 0, 255),
        (255, 0, 0),
        (255, 255, 0),
    ]

    # === 5. 创建输出文件夹 ===
    overlay_dir = os.path.join(outdir, "batch_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    # === 6. 遍历每个测点，绘制截线、速度文本与箭头 ===
    for row in batch_results:
        angle = row.get("angle_probe_deg")
        if angle is None:
            continue  # 无方向则跳过

        idx = int(row.get("index", 0))
        point = (int(row.get("point_x", 0)), int(row.get("point_y", 0)))
        length = int(row.get("length_px", LENGTH_PX))
        slope = row.get("slope_px_per_frame")
        fps_here = row.get("fps") or default_fps
        color = colors[idx % len(colors)]  # 循环取色

        speed_val = row.get("speed_m_per_s")
        overlay_speed = row.get("_overlay_speed_mps")
        speed_for_check = overlay_speed if overlay_speed is not None else speed_val
        out_of_range = _is_speed_out_of_range(speed_for_check)

        # === 6.1 绘制测速截线及点位 ===
        if not out_of_range:
            (x1, y1, x2, y2), _ = _line_endpoints(point, length, angle)
            cv2.line(overview, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            cv2.circle(overview, point, 4, color, -1, cv2.LINE_AA)

        # === 6.3 速度越界时仅标记叉号并跳过文字/箭头绘制 ===
        if out_of_range:
            cross_size = max(6, int(round(length * 0.1)))
            cv2.line(overview, (point[0] - cross_size, point[1] - cross_size),
                     (point[0] + cross_size, point[1] + cross_size), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(overview, (point[0] - cross_size, point[1] + cross_size),
                     (point[0] + cross_size, point[1] - cross_size), (0, 0, 255), 2, cv2.LINE_AA)
            continue

        # === 6.4 绘制文字标签（序号 + 速度） ===
        text = ""
        if overlay_speed is not None:
            text = f" {overlay_speed:.2f} m/s"
        elif speed_val is not None:
            text = f" {speed_val:.2f} m/s"
        else:
            text = "N/A"

        cv2.putText(overview, text, (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # === 6.4 计算箭头长度（按速度比例缩放） ===
        min_arrow_len = max(20, int(round(length * 0.2)))
        max_arrow_len = max(min_arrow_len + 1, int(round(length * 0.7)))
        arrow_len = min_arrow_len
        if overlay_speed is not None and max_speed and max_speed > 0:
            scale = abs(overlay_speed) / max_speed
            arrow_len = int(round(min_arrow_len + scale * (max_arrow_len - min_arrow_len)))

        # === 6.5 确定箭头方向（正负速度） ===
        # 方向严格按 slope 正负决定；只有在没有 slope 时才退回速度符号
        if slope is not None:
            sign = 1 if slope >= 0 else -1
        elif overlay_speed is not None:
            sign = 1 if overlay_speed >= 0 else -1
        else:
            sign = 1

        # === 6.6 画箭头 ===
        _, direction = _line_endpoints(point, 2, angle)  # 单位方向向量
        dx, dy = direction
        start = (int(point[0]), int(point[1]))
        end = (int(point[0] + sign * dx * arrow_len), int(point[1] + sign * dy * arrow_len))
        cv2.arrowedLine(overview, start, end, color, 3, tipLength=_fixed_arrow_tip_ratio(arrow_len))

    # === 7. 保存总览图 ===
    overview_path = os.path.join(overlay_dir, "batch_overview.png")
    cv2.imwrite(overview_path, overview)
    print(f"[batch overlay] 总览图已保存：{overview_path}")


def save_single_summary(
    outdir: str,
    best: Dict[str, object],
    *,
    length_px: int,
    spatial_sample_step: int,
    spatial_sample_count_value: int,
    max_frames: int,
    m_per_px: Optional[float],
) -> None:
    """保存单点测速 summary.csv。"""
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
        "alpha_deg": best.get("angle"),
        "score": best.get("score"),
    }
    fieldnames = list(row.keys())
    os.makedirs(outdir, exist_ok=True)
    summary_path = os.path.join(outdir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(f"[summary] {os.path.abspath(summary_path)}")


def main():
    """程序入口，执行测速流程。"""
    if not os.path.isfile(VIDEO):
        raise FileNotFoundError(f"视频不存在: {VIDEO}")

    outdir = init_debug_dir(tag="stiv-accu-vote")
    process_video = VIDEO
    roi_box: Optional[Tuple[int, int, int, int]] = None
    roi_offset = (0, 0)
    center_proc = CENTER
    bank_point_proc = BANK_POINT
    calib_line_proc = CALIB_LINE_XYXY

    cap0 = cv2.VideoCapture(VIDEO)
    ok0, frame0 = cap0.read()
    cap0.release()
    if not ok0:
        raise RuntimeError(f"无法读取视频首帧: {VIDEO}")
    frame_h, frame_w = frame0.shape[:2]
    spatial_count = spatial_sample_count(LENGTH_PX, SPATIAL_SAMPLE_STEP)

    if ROI is not None:
        roi_box = _normalize_roi(ROI, frame_w, frame_h)
        if not _point_in_roi(CENTER, roi_box):
            raise ValueError(f"CENTER 点落在 ROI 外: CENTER={CENTER}, ROI={roi_box}")

        center_proc = _global_to_local_point(CENTER, roi_box)
        bank_point_proc = _global_to_local_point(BANK_POINT, roi_box)
        calib_line_proc = _global_to_local_line(CALIB_LINE_XYXY, roi_box)
        roi_offset = (roi_box[0], roi_box[1])

        roi_video_path = os.path.join(outdir, "roi_cropped.mp4")
        _crop_video_to_roi(VIDEO, roi_video_path, roi_box)
        process_video = roi_video_path

    check_w = frame_w if roi_box is None else (roi_box[2] - roi_box[0])
    check_h = frame_h if roi_box is None else (roi_box[3] - roi_box[1])
    for angle in _iter_angles(ANGLE_START, ANGLE_END, ANGLE_STEP):
        if not _line_fully_inside_frame(center_proc, LENGTH_PX, angle, check_w, check_h):
            raise ValueError(
                f"CENTER 对应测速线超出ROI: center={CENTER}, angle={angle}, LENGTH_PX={LENGTH_PX}, ROI={roi_box}"
            )

    print(f"[out] 所有步骤图将保存到：{outdir}")
    print(f"[cfg] CENTER={CENTER}, LENGTH_PX={LENGTH_PX}, SPATIAL_SAMPLE_STEP={SPATIAL_SAMPLE_STEP}, "
          f"spatial_sample_count={spatial_count}, ANGLES=({ANGLE_START},{ANGLE_END},{ANGLE_STEP}), "
          f"MAX_FRAMES={MAX_FRAMES}, USE_ROI={USE_ROI}")
    print(
        f"[cfg] VOTE_THETA_RANGE={VOTE_THETA_RANGE}, coarse_to_fine={USE_COARSE_TO_FINE_THETA}, "
        f"coarse_res={VOTE_THETA_COARSE_RES_DEG}, fine_res={VOTE_THETA_FINE_RES_DEG}, "
        f"fine_half_width={VOTE_THETA_FINE_HALF_WIDTH_DEG}"
    )
    print(f"[cfg] ROI={ROI}")
    print(f"[cfg] START_FRAME={START_FRAME}, START_TIME_SEC={START_TIME_SEC}")
    print(f"[cfg] V_MIN={V_MIN}, V_MAX={V_MAX} (单位：m/s，None 表示不限制)")

    if START_FRAME is not None and START_TIME_SEC is not None:
        raise ValueError("START_FRAME 与 START_TIME_SEC 只能设置一个")
    if START_FRAME is not None and START_FRAME < 0:
        raise ValueError("START_FRAME 不能为负数")
    if START_TIME_SEC is not None and START_TIME_SEC < 0:
        raise ValueError("START_TIME_SEC 不能为负数")

    # 计算/确定比例尺
    m_per_px = SCALE_M_PER_PIXEL
    if m_per_px is None and (CALIB_REAL_M is not None and CALIB_LINE_XYXY is not None):
        m_per_px = compute_scale_from_first_frame(VIDEO, CALIB_LINE_XYXY, CALIB_REAL_M)
    if m_per_px is not None:
        print(f"[scale] 使用 SCALE_M_PER_PIXEL={m_per_px:.6f} m/px")
    else:
        print("[scale] 未提供比例尺；将仅输出像素单位的斜率，不计算 m/s")

    if USE_BATCH_LINE_PROBING:
        from stiv_adapt.search import batch_probe_along_line

        results = batch_probe_along_line(
            video_path=process_video,
            center=center_proc,
            bank_point=bank_point_proc,
            interval_px=PROBE_INTERVAL_PX,
            length_px=LENGTH_PX,
            angle_range=(ANGLE_START, ANGLE_END, ANGLE_STEP),
            max_frames=MAX_FRAMES,
            spatial_sample_step=SPATIAL_SAMPLE_STEP,
            start_frame=START_FRAME,
            start_time_sec=START_TIME_SEC,
            m_per_px=m_per_px,
            fps=FPS,
            use_circular_roi=USE_ROI,
            roi_radius_frac=ROI_RADIUS_FRAC,
            edge_method=EDGE_METHOD,
            direction_method=DIRECTION_METHOD,
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
            top_k_candidates=TOP_K_CANDIDATES,
            verbose=VERBOSE,
            coord_offset=roi_offset,
            k_sigma=K_SIGMA,
            score_mode=SCORE_MODE,
            save_debug_images=SAVE_DEBUG_IMAGES,
        )

        print("\n====== 多点测速结果 ======")
        for row in results:
            speed_txt = "N/A" if row["speed_m_per_s"] is None else f"{row['speed_m_per_s']:.4f} m/s"
            print(
                f"#{row['index']:02d} pt=({row['point_x']},{row['point_y']}) "
                f"len={row['length_px']}px step={row['spatial_sample_step']} "
                f"samples={row['spatial_sample_count']} angle={row['angle_probe_deg']}° "
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
            video_path=process_video,
            outdir=outdir,
            center=center_proc,
            bank_point=bank_point_proc,
            batch_results=overlay_results,
            m_per_px=m_per_px,
            default_fps=FPS,
        )
        return


    # 自适应方向搜索（Hough 或自相关；由 DIRECTION_METHOD 控制）
    best = adaptive_direction_search(
        video_path=process_video,
        center=center_proc,
        length_px=LENGTH_PX,
        angle_start=ANGLE_START,
        angle_end=ANGLE_END,
        angle_step=ANGLE_STEP,
        max_frames=MAX_FRAMES,
        spatial_sample_step=SPATIAL_SAMPLE_STEP,
        start_frame=START_FRAME,
        start_time_sec=START_TIME_SEC,
        use_circular_roi=USE_ROI,
        roi_radius_frac=ROI_RADIUS_FRAC,
        edge_method=EDGE_METHOD,
        direction_method=DIRECTION_METHOD,
        verbose=VERBOSE,
        use_E_asym=USE_E_ASYM,
        use_M_mono=USE_M_MONO,
        m_mono_peak_ratio=M_MONO_PEAK_RATIO,
        # —— 将 run 的可调参数传入 search —— #
        vote_theta_res_deg=VOTE_THETA_RES_DEG,
        vote_k_ratio=VOTE_K_RATIO,
        vote_theta_range=VOTE_THETA_RANGE,
        use_coarse_to_fine_theta=USE_COARSE_TO_FINE_THETA,
        vote_theta_coarse_res_deg=VOTE_THETA_COARSE_RES_DEG,
        vote_theta_fine_res_deg=VOTE_THETA_FINE_RES_DEG,
        vote_theta_fine_half_width_deg=VOTE_THETA_FINE_HALF_WIDTH_DEG,
        top_k_candidates=TOP_K_CANDIDATES,
        k_sigma=K_SIGMA,
        score_mode=SCORE_MODE,
        save_all_sti=SAVE_ALL_STI,
        save_debug_images=SAVE_DEBUG_IMAGES,
        #vote_rho_step=VOTE_RHO_STEP,
    )

    # 覆写/确认 FPS
    if FPS is not None:
        best["fps"] = float(FPS)
    if not best.get("fps"):
        print("[warn] 无法可靠获取 FPS；建议在配置区手动设置 FPS。")
    else:
        print(f"[fps] 视频 FPS={best['fps']:.6f}")

    # 叠加到首帧预览图
    slope = best["slope"]      # dx/dy (STI sample/frame)
    save_flow_overlay(
        video_path=process_video,
        outdir=outdir,
        center=center_proc,
        best_angle_deg=best.get("angle_probe", best["angle"]),
        length_px=LENGTH_PX,
        slope_px_per_frame=slope,
        spatial_sample_step=SPATIAL_SAMPLE_STEP,
        spatial_sample_count=spatial_count,
        m_per_px=m_per_px,
        fps=best.get("fps"),
        calib_xyxy=calib_line_proc,
        calib_real_m=CALIB_REAL_M,
        center_display=CENTER,
        filename="frame_overlay.png",
        preview_max_side=1280
    )

    # 打印结果与速度换算
    print("\n====== 最终结果 ======")
    print(f"中心点: {CENTER}")
    print(f"LENGTH_PX: {LENGTH_PX} px")
    print(f"SPATIAL_SAMPLE_STEP: {SPATIAL_SAMPLE_STEP}")
    print(f"spatial_sample_count: {spatial_count}")
    print(f"MAX_FRAMES: {MAX_FRAMES}")
    #print(f"最佳条纹角度: {best['angle']} °")

    print(f"测速线方向: {best.get('angle_probe'):} °")
    print(f"最佳纹理角度α: {best['angle']:} °")

    print(f"测速线评分策略: {best.get('score_mode', SCORE_MODE)}")
    if best.get("peak_votes") is not None:
        print(f"Hough 主峰强度(peak_votes): {best['peak_votes']:.1f}")
    if best.get("peak_ratio") is not None:
        print(f"主峰占比(peak_ratio): {best['peak_ratio']:.4f}")
    print(f"用于筛选的得分: {best['score']:.4f}" if best.get("score_mode") == "peak_ratio" else f"用于筛选的得分: {best['score']:.1f}")
    slope_sti = best.get("slope_sti", best["slope"])
    velocity_px_per_frame = best.get("velocity_px_per_frame_corrected")
    if velocity_px_per_frame is None:
        velocity_px_per_frame = _correct_velocity_px_per_frame(slope_sti, SPATIAL_SAMPLE_STEP)

    print(f"STI 斜率 slope_sti (sample/frame): {slope_sti if slope_sti is not None else 'None'}")
    print(f"校正后像素速度 velocity_px_per_frame_corrected: {velocity_px_per_frame if velocity_px_per_frame is not None else 'None'} px/frame")
    if velocity_px_per_frame is not None and best.get("fps"):
        v_pxps = velocity_px_per_frame * best["fps"]
        print(
            f"像素速度: {v_pxps:.4f} px/s   "
            f"(velocity_px_per_frame={velocity_px_per_frame:.6f}, FPS={best['fps']:.3f})"
        )
    else:
        print("未计算像素速度：缺少 slope_sti 或 FPS。")
    if USE_M_MONO:
        print(f"M_mono 单调性评分: {best.get('M_mono')}")
    if USE_E_ASYM:
        print(f"E_asym 对称性评分: {best.get('E_asym')}")

    v_mps = _velocity_mps(velocity_px_per_frame, m_per_px, best.get("fps"))
    if v_mps is not None:
        print(
            f"速度估计: {v_mps:.4f} m/s   "
            f"(velocity_px_per_frame={velocity_px_per_frame:.6f}, "
            f"m/px={m_per_px:.6f}, FPS={best['fps']:.3f})"
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

    # 耗时统计
    n_lines = best.get("num_lines", 0)
    t_total = best.get("total_time_sec", 0.0)
    times = best.get("angle_times") or []

    print(f"测速线数量: {n_lines}")
    print(f"总用时: {t_total:.3f} s")

    if times:
        avg = sum(t["seconds"] for t in times) / len(times)
        slow = max(times, key=lambda t: t["seconds"])
        print(f"单条平均用时: {avg:.3f} s，最慢: {slow['angle']:.1f}° → {slow['seconds']:.3f} s")

    print("所有步骤图已写入输出目录。")


if __name__ == "__main__":
    main()
    t1 = time.perf_counter()
    print(f"[TIME] total = {t1 - t0:.3f} s")
