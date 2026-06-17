from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

from .sobel import (
    DEFAULT_WEIGHT_MAX,
    DEFAULT_WEIGHT_MIN,
    compute_sobel_edges,
    hough_angle_voting_weighted,
)


@dataclass(frozen=True)
class SingleSTISobelConfig:
    """单张 STI 的 Sobel + weighted Hough 配置。"""

    use_circular_roi: bool = True
    roi_radius_frac: float = 0.9
    theta_res_deg: float = 1.0
    rho_step: float = 1.0
    vote_theta_range: Tuple[float, float] = (0.0, 180.0)
    k_sigma: float = 1.0
    weight_min: float = DEFAULT_WEIGHT_MIN
    weight_max: float = DEFAULT_WEIGHT_MAX
    near_horizontal_tol_deg: float = 1.0


@dataclass(frozen=True)
class SingleSTIBestResult:
    """主峰结果（该 STI 的 best）。"""

    theta_normal_deg: Optional[float]
    angle_deg: Optional[float]
    slope: Optional[float]
    peak_value: float
    peak_ratio: float
    peak_idx: Optional[int]
    sum_filtered_scores: float


@dataclass(frozen=True)
class SingleSTIAnalysisResult:
    """单张 STI 分析结果。

    - theta_axis: 法线角轴（单位：度，范围通常是 [0, 180)）
    - scores: 原始 weighted Hough 角度得分（过滤前）
    - scores_filtered: 角度范围过滤后的得分（仅保留 vote_theta_range）
    - edges: Sobel 路线输出的边缘/纹理增强图（uint8）
    """

    best: SingleSTIBestResult
    edges: np.ndarray
    scores: np.ndarray
    theta_axis: np.ndarray
    scores_filtered: np.ndarray
    total_score_raw: float
    total_score_filtered: float
    rho_max: int
    config: SingleSTISobelConfig


def _to_uint8_sti(sti: np.ndarray) -> np.ndarray:
    """将输入 STI 归一化为单通道 uint8。"""

    if not isinstance(sti, np.ndarray):
        raise TypeError("sti must be numpy.ndarray")
    if sti.ndim != 2:
        raise ValueError("sti must be a 2D grayscale array")

    if sti.dtype == np.uint8:
        return sti

    arr = np.asarray(sti)
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, 255).astype(np.uint8)

    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"unsupported sti dtype: {arr.dtype}")

    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)

    vmin = float(arr[finite].min())
    vmax = float(arr[finite].max())
    if vmax - vmin < 1e-9:
        out = np.zeros(arr.shape, dtype=np.uint8)
        out[finite] = np.clip(arr[finite], 0, 255).astype(np.uint8)
        return out

    norm = np.zeros(arr.shape, dtype=np.float32)
    norm[finite] = (arr[finite] - vmin) / (vmax - vmin)
    return np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)


def _apply_theta_filters_on_scores(
    scores: np.ndarray,
    theta_axis: np.ndarray,
    theta_range: Tuple[float, float],
) -> np.ndarray:
    """按法线角范围过滤得分，逻辑与 search.py 保持一致。"""

    filtered = scores.copy()
    th_min, th_max = theta_range
    valid = (theta_axis >= th_min) & (theta_axis < th_max)
    filtered[~valid] = 0
    return filtered


def _calc_slope_from_alpha(alpha_deg: float, near_horizontal_tol_deg: float) -> Optional[float]:
    """由纹理方向角 alpha 计算 slope=dx/dy；近水平时返回 None。"""

    tol = max(0.0, float(near_horizontal_tol_deg))
    alpha_mod = alpha_deg % 180.0
    dist_to_horizontal = min(alpha_mod, 180.0 - alpha_mod)
    if dist_to_horizontal <= tol:
        return None

    tan_a = math.tan(math.radians(alpha_deg))
    if abs(tan_a) < 1e-9:
        return None
    return 1.0 / tan_a


def _extract_peak(
    scores_filtered: np.ndarray,
    theta_axis: np.ndarray,
    near_horizontal_tol_deg: float,
) -> SingleSTIBestResult:
    """提取主峰并换算纹理方向，语义与 search.py 的 _extract_peak_from_votes 对齐。"""

    if scores_filtered.size == 0:
        return SingleSTIBestResult(
            theta_normal_deg=None,
            angle_deg=None,
            slope=None,
            peak_value=0.0,
            peak_ratio=0.0,
            peak_idx=None,
            sum_filtered_scores=0.0,
        )

    sum_scores = float(scores_filtered.sum())
    if sum_scores <= 0.0:
        return SingleSTIBestResult(
            theta_normal_deg=None,
            angle_deg=None,
            slope=None,
            peak_value=0.0,
            peak_ratio=0.0,
            peak_idx=None,
            sum_filtered_scores=0.0,
        )

    peak_idx = int(np.argmax(scores_filtered))
    theta_normal_deg = float(theta_axis[peak_idx])
    peak_value = float(scores_filtered[peak_idx])
    peak_ratio = peak_value / (sum_scores + 1e-9)
    angle_deg = (theta_normal_deg + 90.0) % 180.0
    slope = _calc_slope_from_alpha(angle_deg, near_horizontal_tol_deg=near_horizontal_tol_deg)

    return SingleSTIBestResult(
        theta_normal_deg=theta_normal_deg,
        angle_deg=angle_deg,
        slope=slope,
        peak_value=peak_value,
        peak_ratio=peak_ratio,
        peak_idx=peak_idx,
        sum_filtered_scores=sum_scores,
    )


def analyze_single_sti_sobel(
    sti: np.ndarray,
    config: SingleSTISobelConfig,
) -> SingleSTIAnalysisResult:
    """分析一张已合成好的 STI，返回主方向与完整角度曲线数据。

    流程固定为：
    1) Sobel 纹理增强（可选圆形 ROI）
    2) weighted Hough 角度投票（mu + k_sigma*sigma 阈值截断）
    3) 法线角范围过滤（仅保留 vote_theta_range）
    4) 提取主峰并换算纹理方向 alpha 与 slope

    注意：
    - 该接口不扫描 angle_probe，不依赖视频帧，不涉及 build_sti_from_frames。
    - best 的判据采用 peak_ratio（同时返回 peak_value 便于调试）。
    """

    sti_u8 = _to_uint8_sti(sti)

    edges = compute_sobel_edges(
        sti_u8,
        use_circular_roi=config.use_circular_roi,
        roi_radius_frac=config.roi_radius_frac,
        save_mag_name=None,
        save_edge_name=None,
        verbose=False,
    )

    total_raw, _, scores, theta_axis, rho_max, _ = hough_angle_voting_weighted(
        edges,
        theta_res_deg=config.theta_res_deg,
        rho_step=config.rho_step,
        weight_min=config.weight_min,
        weight_max=config.weight_max,
        use_circular_roi=config.use_circular_roi,
        roi_radius_frac=config.roi_radius_frac,
        verbose=False,
        k_sigma=config.k_sigma,
    )

    scores_filtered = _apply_theta_filters_on_scores(
        scores,
        theta_axis,
        theta_range=config.vote_theta_range,
    )
    best = _extract_peak(
        scores_filtered,
        theta_axis,
        near_horizontal_tol_deg=config.near_horizontal_tol_deg,
    )

    return SingleSTIAnalysisResult(
        best=best,
        edges=edges,
        scores=scores,
        theta_axis=theta_axis,
        scores_filtered=scores_filtered,
        total_score_raw=float(total_raw),
        total_score_filtered=float(scores_filtered.sum()),
        rho_max=int(rho_max),
        config=config,
    )


__all__ = [
    "SingleSTISobelConfig",
    "SingleSTIBestResult",
    "SingleSTIAnalysisResult",
    "analyze_single_sti_sobel",
]

