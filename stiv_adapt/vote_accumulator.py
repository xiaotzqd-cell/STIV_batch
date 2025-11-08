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
import cv2


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

# ===== 测试入口 =============================================
def quick_test(
    edge_path=None,
    edge_u8: np.ndarray | None = None,
    theta_res_deg: float = 1.0,
    rho_step: float = 1.0,
    k_ratio: float = 0.55,
    save_csv: bool = False,
    verbose: bool = True,
):
    """
    直接在代码里调用的测试入口（不使用命令行）。
    用法1：传 edge_path（单通道Canny图路径）
    用法2：传 edge_u8 （已经是 uint8 的Canny边缘图）

    返回：(total_lines, angle_votes, votes_per_theta, theta_axis, rho_max, best_info)
    """
    if edge_u8 is None:
        if edge_path is None:
            raise ValueError("请提供 edge_path 或 edge_u8 其中之一")
        img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(edge_path)
        edge_u8 = img

    total, angle_votes, votes_full, theta_axis, rho_max, best = hough_angle_voting_min(
        edge_u8,
        theta_res_deg=theta_res_deg,
        rho_step=rho_step,
        k_ratio=k_ratio,
        verbose=verbose,
    )

    if verbose and len(angle_votes) > 0:
        top = sorted(angle_votes, key=lambda x: (-x[1], x[0]))[:10]
        print(f"[RESULT] 角度得分 Top-{len(top)}（θ为法线角；votes=该θ上≥K的ρ-bin个数）:")
        for ang, v in top:
            print(f"  θ = {ang:6.2f}°, votes = {v:5d}")

    if save_csv:
        try:
            import pandas as pd
            arr = np.stack([theta_axis, votes_full], axis=1)
            df = pd.DataFrame(arr, columns=["theta_deg", "votes(>=K rho-bins)"])
            df.to_csv("theta_votes.csv", index=False, float_format="%.6f")
            if verbose:
                print("[RESULT] 已保存 θ-得分 表到: theta_votes.csv")
        except Exception as e:
            if verbose:
                print(f"[WARN] 保存CSV失败：{e}")

    return total, angle_votes, votes_full, theta_axis, rho_max, best


# 可选：直接点“运行”也能跑（不依赖 sys.argv）
if __name__ == "__main__":
    # 默认尝试当前目录下的临时边缘图；你可以改成自己的路径
    default_path = r"D:\Programs\Python\stiv\out\20251027-193952-stiv-accu-vote\step7_canny_edges.png"
    try:
        quick_test(edge_path=default_path, theta_res_deg=1.0, rho_step=1.0, k_ratio=0.55,
                   save_csv=True, verbose=True)
    except Exception as e:
        print(f"[SELFTEST] 无法读取 {default_path}，请在 quick_test(edge_path=...) 里改成你的边缘图路径。错误：{e}")



