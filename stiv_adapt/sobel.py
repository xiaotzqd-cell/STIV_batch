# -*- coding: utf-8 -*-
"""
sobel.py — 使用 Sobel 算子计算 STI 边缘/梯度图。
按照用户提供的实现：可直接计算梯度幅值，或先做大尺度高斯高通后再算梯度。
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict

# 亮度阈值默认值：集中放在顶部，便于统一调整
DEFAULT_WEIGHT_MIN: float = 5.0
DEFAULT_WEIGHT_MAX: float = 255.0


def build_J1_grad_mag(img: np.ndarray) -> np.ndarray:
    """方法1：直接使用梯度幅值 J = |∇I|。"""
    # x 方向梯度
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    # y 方向梯度
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    # 梯度幅值
    mag = cv2.magnitude(gx, gy)
    # 归一化到 0~255
    J1 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return J1.astype(np.uint8)


def build_J2_highpass_grad(img: np.ndarray, sigma: float = 9.0) -> np.ndarray:
    """方法2：先去除慢变化背景，再计算梯度幅值。"""
    # 大尺度高斯模糊作为背景
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # 高频分量（原图减去背景）
    high = img.astype(np.float32) - blur.astype(np.float32)
    # 高频归一化到 0~255，避免幅值过小
    high_norm = cv2.normalize(high, None, 0, 255, cv2.NORM_MINMAX)

    # 在高频图上计算梯度
    gx = cv2.Sobel(high_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(high_norm, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # 再次归一化到 0~255
    J2 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return J2.astype(np.uint8)


def build_circular_roi_mask(shape: tuple[int, int], radius_frac: float = 0.9) -> np.ndarray:
    """
    构造圆形 ROI 掩膜（True 表示在圆内）。
    radius_frac：圆半径占 min(H, W)/2 的比例，范围 (0, 1]。
    例如 radius_frac=0.9 表示半径 = 0.9 * min(H, W)/2。
    """
    H, W = shape
    cx = W / 2.0
    cy = H / 2.0

    yy, xx = np.ogrid[:H, :W]
    dx = xx.astype(np.float32) - cx
    dy = yy.astype(np.float32) - cy
    dist2 = dx * dx + dy * dy

    radius = float(max(0.0, min(1.0, radius_frac))) * 0.5 * float(min(H, W))
    mask = dist2 <= (radius * radius)
    return mask


def hough_angle_score_weighted(J: np.ndarray,
                               theta_min_deg: float = 0.0,
                               theta_max_deg: float = 180.0,
                               theta_step_deg: float = 1.0,
                               rho_step: float = 1.0,
                               weight_min: float = DEFAULT_WEIGHT_MIN,
                               weight_max: float = DEFAULT_WEIGHT_MAX,
                               roi_mask: Optional[np.ndarray] = None
                               ) -> tuple[np.ndarray, np.ndarray]:
    """
    在 J 图像上做“灰度加权”的简化 Hough 角度评分。
    角度范围固定为 [0, 180]（按需求写死，不再外部配置）。
    """
    H, W = J.shape

    J_float = J.astype(np.float32)
    base_mask = (J_float >= weight_min) & (J_float <= weight_max)
    if roi_mask is not None:
        base_mask = base_mask & roi_mask

    ys, xs = np.nonzero(base_mask)
    if len(xs) == 0:
        thetas_deg = np.arange(theta_min_deg,
                               theta_max_deg + 1e-3,
                               theta_step_deg,
                               dtype=np.float32)
        scores = np.zeros_like(thetas_deg, dtype=np.float32)
        return thetas_deg, scores

    weights = J[ys, xs].astype(np.float32)

    cx = W / 2.0
    cy = H / 2.0
    X = xs.astype(np.float32) - cx
    Y = ys.astype(np.float32) - cy

    thetas_deg = np.arange(theta_min_deg,
                           theta_max_deg + 1e-3,
                           theta_step_deg,
                           dtype=np.float32)
    thetas_rad = np.deg2rad(thetas_deg)
    cos_t = np.cos(thetas_rad)
    sin_t = np.sin(thetas_rad)

    rho_max = float(np.hypot(cx, cy))
    rho_bins = int(np.floor((2.0 * rho_max) / rho_step) + 1)
    scores = np.zeros_like(thetas_deg, dtype=np.float32)

    for i, (ct, st) in enumerate(zip(cos_t, sin_t)):
        rho = X * ct + Y * st
        r = (rho + rho_max) / rho_step
        i0 = np.floor(r).astype(np.int32)
        frac = (r - i0).astype(np.float32)

        acc = np.zeros(rho_bins, dtype=np.float32)

        valid0 = (i0 >= 0) & (i0 < rho_bins)
        if np.any(valid0):
            np.add.at(acc, i0[valid0], (1.0 - frac[valid0]) * weights[valid0])

        i1 = i0 + 1
        valid1 = (i1 >= 0) & (i1 < rho_bins)
        if np.any(valid1):
            np.add.at(acc, i1[valid1], frac[valid1] * weights[valid1])

        mu = float(acc.mean())
        sigma = float(acc.std())
        thr = mu + sigma

        above = acc - thr
        above[above < 0.0] = 0.0
        scores[i] = float(above.sum())

    return thetas_deg, scores


def hough_angle_voting_weighted(
    J: np.ndarray,
    theta_res_deg: float = 1.0,
    rho_step: float = 1.0,
    weight_min: float = DEFAULT_WEIGHT_MIN,
    weight_max: float = DEFAULT_WEIGHT_MAX,
    use_circular_roi: bool = False,
    roi_radius_frac: float = 1.0,
    verbose: bool = False,
) -> Tuple[float, List[Tuple[float, float]], np.ndarray, np.ndarray, int, Dict[str, float]]:
    """基于 Sobel 梯度幅值的“灰度加权”角度评分。

    返回与 ``hough_angle_voting_min`` 相同的 6 项元组，便于现有流程复用：
    ``(total_score, angle_votes, scores_per_theta, theta_axis, rho_max, best_info)``。
    其中 ``best_info['votes']`` 为最佳角度上的能量和（非整数票数）。
    """

    if J.ndim != 2:
        raise ValueError("hough_angle_voting_weighted 仅支持单通道灰度图")

    H, W = J.shape
    roi_mask = None
    if use_circular_roi:
        roi_mask = build_circular_roi_mask(J.shape, radius_frac=roi_radius_frac)

    theta_axis, scores = hough_angle_score_weighted(
        J,
        theta_min_deg=0.0,
        theta_max_deg=180.0,
        theta_step_deg=float(theta_res_deg),
        rho_step=float(rho_step),
        weight_min=float(weight_min),
        weight_max=float(weight_max),
        roi_mask=roi_mask,
    )

    rho_max = int(np.ceil(np.hypot(W / 2.0, H / 2.0)))

    if scores.size == 0:
        zero = np.zeros_like(theta_axis, dtype=np.float32)
        best_info = {"theta_deg": float("nan"), "alpha_deg": float("nan"), "votes": 0.0}
        return 0.0, [], zero, theta_axis, rho_max, best_info

    best_idx = int(np.argmax(scores))
    theta_best = float(theta_axis[best_idx])
    alpha_best = (theta_best + 90.0) % 180.0
    votes_best = float(scores[best_idx])

    best_info = {"theta_deg": theta_best, "alpha_deg": alpha_best, "votes": votes_best}

    total_score = float(scores.sum())
    angle_votes = [(float(theta_axis[i]), float(v)) for i, v in enumerate(scores) if v > 0]

    if verbose:
        rho_bins = int(np.floor((2 * rho_max) / rho_step) + 1)
        print(
            f"[RESULT] (H×W)={H}×{W} | theta_res_deg={theta_res_deg} | rho_step={rho_step} | "
            f"weight_min={weight_min} | weight_max={weight_max}"
        )
        print(f"[RESULT] ρ_max={rho_max} | ρ_bins={rho_bins}")
        print(
            f"[RESULT] φ* (theta_deg)={theta_best:.3f} | α*=φ*+90°={alpha_best:.3f} | "
            f"energy={votes_best:.1f}"
        )

    return total_score, angle_votes, scores, theta_axis, rho_max, best_info


def _save_img_safe(name: str, img: np.ndarray) -> None:
    """延迟导入 core._save_img，避免循环依赖；失败则直接写入当前目录。"""
    try:
        from .core import _save_img  # type: ignore
        _save_img(name, img)
        return
    except Exception:
        pass
    cv2.imwrite(name, img)


def compute_sobel_edges(sti_u8: np.ndarray,
                        use_highpass: bool = False,
                        use_circular_roi: bool = False,
                        roi_radius_frac: float = 1.0,
                        save_mag_name: str = "step6_sobel_mag.png",
                        save_edge_name: str = "step7_sobel_edges.png",
                        verbose: bool = False) -> np.ndarray:
    """
    计算 Sobel 梯度幅值图；可选择直接梯度（J1）或高通后梯度（J2）。
    返回的 edges 为 0~255 的 8 位图，可直接进入后续霍夫统计。
    """
    if sti_u8.ndim != 2:
        raise ValueError("Sobel 仅支持单通道灰度图")

    if use_highpass:
        mag = build_J2_highpass_grad(sti_u8)
    else:
        mag = build_J1_grad_mag(sti_u8)

    # ROI 掩膜：可选，将圆外区域置零
    if use_circular_roi:
        mask = build_circular_roi_mask(mag.shape, radius_frac=roi_radius_frac)
        edges = mag.copy()
        edges[~mask] = 0
    else:
        edges = mag

    _save_img_safe(save_mag_name, mag)
    _save_img_safe(save_edge_name, edges)

    if verbose:
        method = "J2 高通+梯度" if use_highpass else "J1 直接梯度"
        roi_info = f"roi=圆形(r={roi_radius_frac:.2f})" if use_circular_roi else "roi=无"
        print(f"[sobel] {method}, {roi_info}, 输出已保存")

    return edges
