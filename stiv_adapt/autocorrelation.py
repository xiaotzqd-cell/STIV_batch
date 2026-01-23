from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from . import sobel  # 同目录 sobel.py（你原测流代码）


@dataclass
class AutoCorrConfig:
    subtract_mean: bool = True
    normalize_center: bool = True
    r_min: int = 3
    r_max: int = 80
    n_theta: int = 180
    baseline_mode: str = "median"   # min / median / bottomq
    bottom_q: float = 0.2
    eps: float = 1e-12

    # ——与原测流 sobel 口径对齐的预处理选项——
    use_sobel_mag: bool = False         # True: autocorr 输入用 J（sobel.py 生成）
    use_circular_roi: bool = False      # True: 对 J 应用圆形 ROI
    roi_radius_frac: float = 1.0        # ROI 半径比例


def _prepare_input_image(sti: np.ndarray, cfg: AutoCorrConfig) -> np.ndarray:
    """
    生成 autocorr 输入图 img（float64）：
    - cfg.use_sobel_mag=False：直接用 STI（float64）
    - cfg.use_sobel_mag=True ：用 sobel.py 生成 J（uint8,0..255）再转 float64
    """
    if sti.ndim != 2:
        raise ValueError("STI 必须是单通道 2D 灰度图 (H,W)")

    if not cfg.use_sobel_mag:
        return sti.astype(np.float64, copy=False)

    # 确保输入 sobel.py 的是 uint8
    sti_u8 = sti.astype(np.uint8, copy=False)

    J = sobel.build_J1_grad_mag(sti_u8)          # uint8 0..255（口径与原测流一致）

    if cfg.use_circular_roi:
        mask = sobel.build_circular_roi_mask(J.shape, radius_frac=float(cfg.roi_radius_frac))
        J = J.copy()
        J[~mask] = 0

    return J.astype(np.float64, copy=False)


def autocorr2d_fft(img: np.ndarray, cfg: AutoCorrConfig) -> np.ndarray:
    """
    计算二维自相关（fftshift 后，零滞后在中心）。
    输入 img: 2D float32/float64
    输出 R: 2D float64
    """
    x = img.astype(np.float64, copy=False)

    if cfg.subtract_mean:
        x = x - np.mean(x)

    F = np.fft.fft2(x)
    P = F * np.conj(F)
    R = np.fft.ifft2(P).real
    R = np.fft.fftshift(R)

    if cfg.normalize_center:
        cy = R.shape[0] // 2
        cx = R.shape[1] // 2
        cval = R[cy, cx]
        if abs(cval) > cfg.eps:
            R = R / cval

    return R


_CACHE = {}


def angular_response_mu_fast(R: np.ndarray, cfg: AutoCorrConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    在自相关图 R 上取环带 (r_min~r_max)，按 θ ∈ [0,180) 分桶，计算每个桶的均值 μ(θ)。
    """
    H, W = R.shape
    key = (H, W, int(cfg.r_min), int(cfg.r_max), int(cfg.n_theta))

    if key not in _CACHE:
        cy, cx = H // 2, W // 2
        yy, xx = np.mgrid[0:H, 0:W]
        dx = (xx - cx).astype(np.float32)
        dy = (yy - cy).astype(np.float32)

        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)        # [-pi, pi]
        theta = np.mod(theta, np.pi)      # [0, pi)

        ring = (r >= cfg.r_min) & (r <= cfg.r_max)

        n = int(cfg.n_theta)
        bin_idx_full = np.floor(theta / np.pi * n).astype(np.int32)
        bin_idx_full = np.clip(bin_idx_full, 0, n - 1)

        bin_idx = bin_idx_full[ring].ravel()
        theta_centers_deg = (np.arange(n) + 0.5) * (180.0 / n)

        _CACHE[key] = (ring, bin_idx, theta_centers_deg)

    ring, bin_idx, theta_deg = _CACHE[key]

    vals = R[ring].astype(np.float64, copy=False).ravel()
    n = int(cfg.n_theta)

    sum_per_bin = np.bincount(bin_idx, weights=vals, minlength=n)
    cnt_per_bin = np.bincount(bin_idx, minlength=n)

    mu = sum_per_bin / np.maximum(cnt_per_bin, 1)
    return theta_deg.astype(np.float64), mu


def _compute_baseline(mu: np.ndarray, cfg: AutoCorrConfig) -> float:
    """按配置计算 mu 的基线值。"""
    if cfg.baseline_mode == "min":
        return float(np.min(mu))
    if cfg.baseline_mode == "median":
        return float(np.median(mu))
    if cfg.baseline_mode == "bottomq":
        q = float(cfg.bottom_q)
        q = min(max(q, 0.0), 1.0)
        m_sorted = np.sort(mu)
        k = max(1, int(len(mu) * q))
        return float(np.mean(m_sorted[:k]))
    return float(np.median(mu))


def orientation_concentration(theta_deg: np.ndarray, mu: np.ndarray, cfg: AutoCorrConfig) -> float:
    """
    180° 周期方向集中度：
      C = | Σ p(θ) * exp(i*2θ) |
    p(θ) 由 mu(θ) 去基线后截断为非负并归一化得到。
    """
    b = _compute_baseline(mu, cfg)
    w = np.maximum(mu - b, 0.0)

    s = float(np.sum(w))
    if s <= cfg.eps:
        return 0.0

    p = w / s
    theta = np.deg2rad(theta_deg)

    c = float(np.sum(p * np.cos(2.0 * theta)))
    s2 = float(np.sum(p * np.sin(2.0 * theta)))
    C = float(np.sqrt(c * c + s2 * s2))

    if C < 0.0:
        C = 0.0
    if C > 1.0:
        C = 1.0
    return C


def compute_autocorr_and_concentration(sti: np.ndarray, cfg: AutoCorrConfig) -> Dict[str, object]:
    """
    一次性输出：
    - R: 自相关图
    - theta_deg: 角度数组
    - mu: 角向响应
    - C: 方向集中度
    """
    img = _prepare_input_image(sti, cfg)

    R = autocorr2d_fft(img, cfg)
    theta_deg, mu = angular_response_mu_fast(R, cfg)
    C = orientation_concentration(theta_deg, mu, cfg)

    return {"R": R, "theta_deg": theta_deg, "mu": mu, "C": C}
