# -*- coding: utf-8 -*-
"""
core.py — STI 的构建与 Canny+Hough 评分
每个关键步骤都会保存调试图片到 DEBUG_RUN_DIR。
"""
import os, math, time
import sys
import pathlib
from typing import Tuple, List, Optional, Dict
from contextlib import contextmanager

import cv2
import numpy as np

if __package__ in (None, ""):
    # 允许直接运行本文件时找到顶层 stiv_adapt 包
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from stiv_adapt.vote_accumulator import hough_angle_voting_min  # 绝对导入，兼容脚本直接运行

# === 输出目录管理 ===
DEBUG_RUN_DIR: Optional[str] = None
def _ensure_dir(p: str):
    """确保目录存在。"""
    os.makedirs(p, exist_ok=True)
def init_debug_dir(base: str = "out", tag: str = "") -> str:
    """初始化调试输出目录并返回路径。"""
    global DEBUG_RUN_DIR
    t = time.strftime("%Y%m%d-%H%M%S")
    name = f"{t}{('-' + tag) if tag else ''}"
    DEBUG_RUN_DIR = os.path.join(base, name)
    _ensure_dir(DEBUG_RUN_DIR)
    return DEBUG_RUN_DIR


@contextmanager
def push_debug_dir(suffix: str):
    """Temporarily descend into a sub-directory under the current DEBUG_RUN_DIR."""
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
    """保存调试图片并返回保存路径。"""
    if DEBUG_RUN_DIR is None:
        init_debug_dir()
    path = os.path.join(DEBUG_RUN_DIR, name)
    _ensure_dir(os.path.dirname(path))
    if img.dtype == np.float32 or img.dtype == np.float64:
        mn, mx = float(img.min()), float(img.max())
        if mx - mn < 1e-9:
            vis = np.zeros_like(img, dtype=np.uint8)
        else:
            vis = np.clip((img - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)
    elif img.dtype == np.uint16:
        vis = np.clip(img / 256.0, 0, 255).astype(np.uint8)
    else:
        vis = img
    cv2.imwrite(path, vis)
    print(f"[save] {os.path.abspath(path)}")
    return path

# === STI 构建 ===
def spatial_sample_count(length_px: int, spatial_sample_step: int = 1) -> int:
    """Return the number of STI spatial samples after line-direction sampling."""
    length_px = int(length_px)
    spatial_sample_step = int(spatial_sample_step)
    if length_px <= 0:
        raise ValueError("length_px 必须为正数")
    if spatial_sample_step <= 0:
        raise ValueError("SPATIAL_SAMPLE_STEP 必须为正数")
    if length_px % spatial_sample_step != 0:
        raise ValueError(
            f"LENGTH_PX={length_px} 不能被 SPATIAL_SAMPLE_STEP={spatial_sample_step} 整除，"
            "请调整参数以避免 STI 空间维度混乱"
        )
    return length_px // spatial_sample_step


def _line_sample_maps(center: Tuple[int, int],
                      length_px: int,
                      angle_deg: float,
                      spatial_sample_step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """生成采样线的 remap 坐标映射。"""
    cx, cy = center
    half = length_px / 2.0
    theta = math.radians(angle_deg)
    dx = math.cos(theta); dy = math.sin(theta)
    sample_count = spatial_sample_count(length_px, spatial_sample_step)
    if int(spatial_sample_step) == 1:
        xs = np.linspace(cx - half * dx, cx + half * dx, num=length_px, dtype=np.float32)
        ys = np.linspace(cy - half * dy, cy + half * dy, num=length_px, dtype=np.float32)
    else:
        offsets = (np.arange(sample_count, dtype=np.float32) - sample_count / 2.0) * float(spatial_sample_step)
        xs = (cx + offsets * dx).astype(np.float32)
        ys = (cy + offsets * dy).astype(np.float32)
    map_x = xs.reshape(1, sample_count); map_y = ys.reshape(1, sample_count)
    return map_x, map_y

def build_sti_from_frames(frames_gray: List[np.ndarray], center: Tuple[int, int],
                          length_px: int, angle_deg: float,
                          spatial_sample_step: int = 1) -> Optional[np.ndarray]:
    """从帧序列构建 STI 图像。"""
    if len(frames_gray) == 0: return None
    map_x, map_y = _line_sample_maps(center, length_px, angle_deg, spatial_sample_step)
    rows = []
    for g in frames_gray:
        row = cv2.remap(g, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rows.append(row)
    sti = np.vstack(rows)  # (T, W)
    return np.clip(sti, 0, 255).astype(np.uint8)

def _angdiff_deg(a, b):
    """计算两个角度的最小差值（度）。"""
    d = abs(a - b)
    return min(d, 180.0 - d)

def _circular_roi_mask(shape: tuple[int, int], radius_frac: float = 1.0) -> np.ndarray:
    """Build a circular ROI mask with configurable radius fraction."""

    H, W = shape
    radius_frac = float(max(0.0, min(1.0, radius_frac)))
    cx, cy = W / 2.0, H / 2.0
    yy, xx = np.indices((H, W))
    r = radius_frac * min(H, W) / 2.0
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * r)


def compute_canny_edges(sti_u8: np.ndarray,
                        use_circular_roi: bool = False,
                        roi_radius_frac: float = 1.0,
                        save_name: Optional[str] = "step7_canny_edges.png",
                        pre_canny_save_name: Optional[str] = "step6_pre_canny_eq_blur.png",
                        verbose: bool = False) -> np.ndarray:
    """对 STI 图像执行 Canny 边缘检测。"""
    H, W = sti_u8.shape[:2]
    eq   = cv2.equalizeHist(sti_u8)
    blur = cv2.GaussianBlur(eq, (5, 5), 0)
    if pre_canny_save_name:
        _save_img(pre_canny_save_name, blur)
    v    = float(np.median(blur))
    low  = int(max(0,   0.66 * v))
    high = int(min(255, 1.33 * v))
    edges = cv2.Canny(blur, low, high, apertureSize=3, L2gradient=True)
    if use_circular_roi:
        mask = _circular_roi_mask((H, W), radius_frac=roi_radius_frac)
        edges = cv2.bitwise_and(edges, edges, mask=mask.astype(np.uint8))
    if save_name:
        _save_img(save_name, edges)
    if verbose:
        print(f"[canny] v={v:.2f}, low={low}, high={high}, roi={'circle' if use_circular_roi else 'none'}")
    return edges


def hough_voting_angle_and_slope(sti_u8: np.ndarray,
                                 edges: np.ndarray,
                                 theta_res_deg: float = 0.5,
                                 rho_step: float = 1.0,
                                 k_ratio: float = 0.55,
                                 save_name: str = "step8_hough_overlay.png",
                                 verbose: bool = False) -> Tuple[float, Optional[float], Optional[float]]:
    """
    返回：(score, slope, angle_deg_line)
      - score: 主峰票数（此处为每 θ 的“≥K 的 ρ-bin 个数”中的最大值）
      - slope: dx/dy；近似水平线→0，近似竖直→inf（返回 None）
      - angle_deg_line: 线方向角（度，0~180）
    """
    import math
    H, W = sti_u8.shape[:2]
    cx, cy = W / 2.0, H / 2.0

    # ==== 这里改成 6 项解包 ====
    total, angle_votes, votes_full, theta_axis, _, _ = hough_angle_voting_min(
        edges,
        theta_res_deg=theta_res_deg,
        rho_step=rho_step,
        k_ratio=k_ratio,
    )

    if votes_full is None or len(votes_full) == 0:
        vis = cv2.cvtColor(sti_u8, cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, "no votes", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)
        _save_img(save_name, vis)
        if verbose: print("[投票] 无有效投票")
        return 0.0, None, None

    peak_idx = int(np.argmax(votes_full))
    theta_normal_deg = float(theta_axis[peak_idx])
    score = float(votes_full[peak_idx])  # = φ*_lines

    alpha_deg = (theta_normal_deg + 90.0) % 180.0
    alpha_rad = math.radians(alpha_deg)
    tan_a = math.tan(alpha_rad)
    slope = None if abs(tan_a) < 1e-9 else (1.0 / tan_a)

    vis = cv2.cvtColor(sti_u8, cv2.COLOR_GRAY2BGR)
    L = np.hypot(H, W)
    ux, uy = math.cos(alpha_rad), math.sin(alpha_rad)
    x1 = int(round(cx - L * ux)); y1 = int(round(cy - L * uy))
    x2 = int(round(cx + L * ux)); y2 = int(round(cy + L * uy))
    cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis,
                f"theta_n={theta_normal_deg:.2f}deg, line={alpha_deg:.2f}deg, "
                f"slope={('None' if slope is None else f'{slope:.4f}')}, peak={score:.0f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv2.LINE_AA)
    _save_img(save_name, vis)

    if verbose:
        print(f"[投票] 峰值 theta_normal={theta_normal_deg:.2f} deg, line_dir={alpha_deg:.2f} deg, "
              f"slope(px/frame)={('None' if slope is None else f'{slope:.6f}')}, peak={score:.0f}")
    return score, slope, alpha_deg
