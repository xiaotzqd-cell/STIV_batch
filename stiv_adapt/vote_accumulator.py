# -*- coding: utf-8 -*-
"""
vote_accumulator.py — 角度投票统计（论文口径，固定双线性入桶）
返回 6 项：
  total_lines: int
  angle_votes: list[(theta_deg, votes)>0]
  votes_per_theta: np.ndarray[int]         # 每个 θ 的“≥K 的 ρ-bin 个数”
  theta_axis: np.ndarray[float] (deg)
  rho_max: int
  best_info: dict {'theta_deg','alpha_deg','votes}  # φ*、α*、与 φ*_lines
"""

from typing import Tuple, List, Dict
import numpy as np


def hough_angle_voting_min(
    edge_u8: np.ndarray,
    theta_res_deg: float = 1.0,
    rho_step: float = 1.0,
    k_ratio: float = 0.55,
    verbose: bool = True,
) -> Tuple[int, List[Tuple[float, int]], np.ndarray, np.ndarray, int, Dict[str, float]]:
    assert edge_u8.ndim == 2 and edge_u8.dtype == np.uint8, "edge image must be single-channel uint8"
    H, W = edge_u8.shape

    # 角度轴（法线角）
    theta_axis = np.arange(0.0, 180.0, float(theta_res_deg), dtype=np.float32)
    thetas = np.deg2rad(theta_axis)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # 边缘点坐标
    yy, xx = np.nonzero(edge_u8)
    if xx.size == 0:
        zero = np.zeros_like(theta_axis, dtype=np.int32)
        best_info = {'theta_deg': float('nan'), 'alpha_deg': float('nan'), 'votes': 0.0}
        return 0, [], zero, theta_axis, 0, best_info

    # 以图像中心为原点
    cx, cy = W / 2.0, H / 2.0
    X = xx.astype(np.float32) - cx
    Y = yy.astype(np.float32) - cy

    # ρ 轴离散
    rho_max = int(np.ceil(np.hypot(cx, cy)))
    rho_bins = int(np.floor((2 * rho_max) / rho_step) + 1)

    # 阈值 K（像素条数门槛）
    r = min(cx, cy)
    K = int(max(1, round(k_ratio * r)))

    votes_per_theta = np.zeros(theta_axis.shape, dtype=np.int32)

    # —— 固定：双线性入桶（分权到相邻两个 ρ-bin）——
    for i, (c, s) in enumerate(zip(cos_t, sin_t)):
        rho = X * c + Y * s
        rcont = (rho + rho_max) / rho_step          # 连续索引
        i0 = np.floor(rcont).astype(np.int32)       # 左桶
        w  = (rcont - i0).astype(np.float32)        # 右桶权重 ∈ [0,1)

        acc = np.zeros(rho_bins, dtype=np.float32)

        # 左桶
        valid0 = (i0 >= 0) & (i0 < rho_bins)
        if np.any(valid0):
            np.add.at(acc, i0[valid0], 1.0 - w[valid0])

        # 右桶
        i1 = i0 + 1
        valid1 = (i1 >= 0) & (i1 < rho_bins)
        if np.any(valid1):
            np.add.at(acc, i1[valid1], w[valid1])

        # 角度得分：该 θ 上“≥K 的 ρ-bin 个数”
        votes_per_theta[i] = int(np.sum(acc >= K))

    total_lines = int(np.sum(votes_per_theta))
    angle_votes = [(float(theta_axis[i]), int(v))
                   for i, v in enumerate(votes_per_theta) if v > 0]

    if angle_votes:
        best_idx = int(np.argmax(votes_per_theta))
        theta_best = float(theta_axis[best_idx])      # φ*（法线角，度）
        lines_best = int(votes_per_theta[best_idx])   # φ*_{lines}（交线频数）
    else:
        theta_best, lines_best = float('nan'), 0

    alpha_best = (theta_best + 90.0) % 180.0          # 条纹/流线方向（度）
    best_info = {'theta_deg': theta_best, 'alpha_deg': alpha_best, 'votes': float(lines_best)}

    if verbose:
        print(f"[RESULT] (H×W)={H}×{W} | theta_res_deg={theta_res_deg} | rho_step={rho_step} | k_ratio={k_ratio}")
        print(f"[RESULT] ρ_max={rho_max} | ρ_bins={rho_bins} | K={K}")
        print(f"[RESULT] φ* (theta_deg)={theta_best:.3f} | α*=φ*+90°={alpha_best:.3f} | φ*_lines={lines_best}")
        print(f"[RESULT] total_lines(sum over θ of ≥K ρ-bins)={total_lines}")

    return total_lines, angle_votes, votes_per_theta, theta_axis, rho_max, best_info

