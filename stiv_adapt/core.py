# -*- coding: utf-8 -*-
"""
core.py — STI 的构建、FFT 扇形滤波增强、以及 Canny+Hough 评分
每个关键步骤都会保存调试图片到 DEBUG_RUN_DIR。
"""
import os, math, time
from typing import Tuple, List, Optional, Dict
import cv2
import numpy as np
from typing import Tuple, Optional
from .vote_accumulator import hough_angle_voting_min  # 相对导入

# === 输出目录管理 ===
DEBUG_RUN_DIR: Optional[str] = None
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
def init_debug_dir(base: str = "out", tag: str = "") -> str:
    global DEBUG_RUN_DIR
    t = time.strftime("%Y%m%d-%H%M%S")
    name = f"{t}{('-' + tag) if tag else ''}"
    DEBUG_RUN_DIR = os.path.join(base, name)
    _ensure_dir(DEBUG_RUN_DIR)
    return DEBUG_RUN_DIR
def _save_img(name: str, img: np.ndarray) -> str:
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
def _line_sample_maps(center: Tuple[int, int], length_px: int, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    cx, cy = center
    half = length_px / 2.0
    theta = math.radians(angle_deg)
    dx = math.cos(theta); dy = math.sin(theta)
    xs = np.linspace(cx - half * dx, cx + half * dx, num=length_px, dtype=np.float32)
    ys = np.linspace(cy - half * dy, cy + half * dy, num=length_px, dtype=np.float32)
    map_x = xs.reshape(1, length_px); map_y = ys.reshape(1, length_px)
    return map_x, map_y

def build_sti_from_frames(frames_gray: List[np.ndarray], center: Tuple[int, int],
                          length_px: int, angle_deg: float) -> Optional[np.ndarray]:
    if len(frames_gray) == 0: return None
    map_x, map_y = _line_sample_maps(center, length_px, angle_deg)
    rows = []
    for g in frames_gray:
        row = cv2.remap(g, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rows.append(row)
    sti = np.vstack(rows)  # (T, W)
    return np.clip(sti, 0, 255).astype(np.uint8)

# === 频域增强 ===
def apply_hann_to_sti(sti_u8: np.ndarray) -> np.ndarray:
    assert sti_u8.ndim == 2
    H, W = sti_u8.shape
    f = sti_u8.astype(np.float32) / 255.0
    wy = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(H) / max(H - 1, 1)))
    wx = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(W) / max(W - 1, 1)))
    win2d = np.outer(wy, wx).astype(np.float32)
    win_img = f * win2d
    _save_img("step2_hann_windowed.png", win_img)
    return win_img

def fft2_on_sti(win32: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    F = np.fft.fft2(win32); F_shift = np.fft.fftshift(F)
    mag = np.abs(F_shift).astype(np.float32)
    mag_log = np.log1p(mag)
    mag_u8 = np.clip(mag_log / mag_log.max() * 255, 0, 255).astype(np.uint8)
    _save_img("step3_fft_magnitude.png", mag_u8)
    return F_shift.astype(np.complex64), mag

def _polar_energy_max_angle(mag: np.ndarray, num_angles: int = 180,
                            rmin_ratio: float = 0.05, rmax_ratio: float = 1.0) -> float:
    H, W = mag.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    rmax = 0.5 * min(H, W) * rmax_ratio
    rmin = 0.5 * min(H, W) * rmin_ratio
    yy, xx = np.indices(mag.shape)
    dy = yy - cy; dx = xx - cx
    rr = np.hypot(dy, dx)
    ang = (np.rad2deg(np.arctan2(dy, dx)) + 180.0) % 180.0
    mask = (rr >= rmin) & (rr <= rmax)
    hist_bins = np.linspace(0.0, 180.0, num_angles + 1, endpoint=True)
    hist, _ = np.histogram(ang[mask], bins=hist_bins, weights=mag[mask])
    k = int(hist.argmax())
    a0 = (hist_bins[k] + hist_bins[k + 1]) * 0.5
    mag_log = np.log1p(mag)
    vis = (mag_log / mag_log.max() * 255.0).astype(np.uint8)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    length = int(0.45 * min(H, W))
    rad = math.radians(a0)
    x1 = int(cx - length * math.cos(rad)); y1 = int(cy - length * math.sin(rad))
    x2 = int(cx + length * math.cos(rad)); y2 = int(cy + length * math.sin(rad))
    cv2.line(vis_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis_bgr, f"theta_fft={a0:.1f} deg", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    _save_img("step3b_fft_with_main_direction.png", vis_bgr)
    return float(a0)

def make_fan_mask(shape: Tuple[int, int], center: Tuple[float, float],
                  angle_deg: float, half_width_deg: float = 4.0,
                  rmin_ratio: float = 0.05, rmax_ratio: float = 1.0) -> np.ndarray:
    H, W = shape
    cy, cx = center
    yy, xx = np.indices((H, W))
    dy = yy - cy; dx = xx - cx
    rr = np.hypot(dy, dx)
    ang = (np.rad2deg(np.arctan2(dy, dx)) + 180.0) % 180.0
    da = np.abs(ang - angle_deg); da = np.minimum(da, 180.0 - da)
    rmax = 0.5 * min(H, W) * rmax_ratio
    rmin = 0.5 * min(H, W) * rmin_ratio
    mask = (da <= half_width_deg) & (rr >= rmin) & (rr <= rmax)
    mask_f32 = mask.astype(np.float32)
    _save_img("step4_fan_mask.png", mask_f32)
    return mask_f32

def ifft_to_spatial(F_shift: np.ndarray) -> np.ndarray:
    F = np.fft.ifftshift(F_shift)
    f = np.fft.ifft2(F)
    real = np.real(f).astype(np.float32)
    u8 = np.zeros_like(real, dtype=np.uint8)
    mn, mx = float(real.min()), float(real.max())
    if mx - mn > 1e-9:
        u8 = np.clip((real - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)
    _save_img("step6_sti_filtered.png", u8)
    return u8

def enhance_sti_via_fft_fan(sti_u8: np.ndarray,
                            half_width_deg: float = 4.0,
                            rmin_ratio: float = 0.05,
                            rmax_ratio: float = 1.0) -> Tuple[np.ndarray, float]:
    _save_img("step1_sti_raw.png", sti_u8)
    win = apply_hann_to_sti(sti_u8)
    F_shift, mag = fft2_on_sti(win)
    theta_fft = _polar_energy_max_angle(mag, num_angles=180, rmin_ratio=rmin_ratio, rmax_ratio=rmax_ratio)
    center = ((mag.shape[0] - 1) / 2.0, (mag.shape[1] - 1) / 2.0)
    mask = make_fan_mask(mag.shape, center, theta_fft, half_width_deg=half_width_deg,
                         rmin_ratio=rmin_ratio, rmax_ratio=rmax_ratio)
    F_filt = F_shift * mask
    mag_masked = np.abs(F_filt).astype(np.float32)
    _save_img("step5_fft_magnitude_masked.png", np.log1p(mag_masked))
    filtered = ifft_to_spatial(F_filt)
    return filtered, theta_fft

def _angdiff_deg(a, b):
    d = abs(a - b)
    return min(d, 180.0 - d)

def compute_canny_edges(sti_u8: np.ndarray,
                        use_circular_roi: bool = False,
                        save_name: str = "step7_canny_edges.png",
                        verbose: bool = False) -> np.ndarray:
    H, W = sti_u8.shape[:2]
    eq   = cv2.equalizeHist(sti_u8)
    blur = cv2.GaussianBlur(eq, (5, 5), 0)
    v    = float(np.median(blur))
    low  = int(max(0,   0.66 * v))
    high = int(min(255, 1.33 * v))
    edges = cv2.Canny(blur, low, high, apertureSize=3, L2gradient=True)
    if use_circular_roi:
        yy, xx = np.indices((H, W))
        cy, cx = H / 2.0, W / 2.0
        r = min(H, W) / 2.0
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * r)
        edges = cv2.bitwise_and(edges, edges, mask=mask.astype(np.uint8))
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
        if verbose: print("[voting] empty votes")
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
                f"theta_n={theta_normal_deg:.1f}deg, line={alpha_deg:.1f}deg, "
                f"slope={('None' if slope is None else f'{slope:.4f}')}, peak={score:.0f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv2.LINE_AA)
    _save_img(save_name, vis)

    if verbose:
        print(f"[voting] peak θ(normal)={theta_normal_deg:.2f}°, line_dir={alpha_deg:.2f}°, "
              f"slope(px/frame)={('None' if slope is None else f'{slope:.6f}')}, peak={score:.0f}")
    return score, slope, alpha_deg
