# -*- coding: utf-8 -*-
"""
search.py — 自适应方向搜索（Canny + 角度投票霍夫）
"""
from typing import Tuple, Optional, Dict, Any, List
import cv2
import numpy as np
import math, time
import csv,os
import pandas as pd
from .core import (
    init_debug_dir,
    build_sti_from_frames,
    enhance_sti_via_fft_fan,
    compute_canny_edges,
)
from .vote_accumulator import hough_angle_voting_min
from stiv_adapt.core import build_sti_from_frames, enhance_sti_via_fft_fan, compute_canny_edges, hough_angle_voting_min

vote_rho_step = 1

def _apply_theta_filters_on_votes(votes_full: np.ndarray,
                                  theta_axis: np.ndarray,
                                  exclude_normals: List[float],
                                  exclude_tol_deg: float,
                                  theta_range: Tuple[float, float]) -> np.ndarray:
    """对 votes_full 施加角度屏蔽与范围裁剪。"""
    vf = votes_full.copy()
    th_min, th_max = theta_range
    valid = (theta_axis >= th_min) & (theta_axis < th_max)
    vf[~valid] = 0
    if exclude_normals and exclude_tol_deg >= 0:
        for ang in exclude_normals:
            # 距离最近等效角（周期 180）
            dist = np.abs(((theta_axis - ang + 90.0) % 180.0) - 90.0)
            mask = dist <= exclude_tol_deg
            vf[mask] = 0
    return vf


def _draw_line_overlay(sti_u8: np.ndarray,
                       alpha_deg: float,
                       theta_normal_deg: float,
                       slope: Optional[float],
                       peak_votes: float,
                       save_name: str) -> None:
    """在 STI 上叠加最佳线方向与文本说明，并保存。"""
    H, W = sti_u8.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    vis = cv2.cvtColor(sti_u8, cv2.COLOR_GRAY2BGR)
    L = float(np.hypot(H, W))
    rad = math.radians(alpha_deg)
    ux, uy = math.cos(rad), math.sin(rad)
    x1 = int(round(cx - L * ux)); y1 = int(round(cy - L * uy))
    x2 = int(round(cx + L * ux)); y2 = int(round(cy + L * uy))
    cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
    text = (f"theta_n={theta_normal_deg:.1f}deg, line={alpha_deg:.1f}deg, "
            f"slope={('None' if slope is None else f'{slope:.4f}')}, peak={peak_votes:.0f}")
    cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(save_name, vis)


def adaptive_direction_search(video_path: str,
                              center: Tuple[int, int],
                              length_px: int,
                              angle_start: float, angle_end: float, angle_step: float,
                              max_frames: int = 300,
                              use_circular_roi: bool = False,
                              use_fft_fan_filter: bool = True,
                              fft_half_width_deg: float = 4.0,
                              fft_rmin_ratio: float = 0.05,
                              fft_rmax_ratio: float = 1.0,
                              verbose: bool = False,
                              vote_theta_res_deg: float = 0.5,
                              vote_k_ratio: float = 0.55,
                              vote_exclude_normals: Optional[List[float]] = None,
                              vote_exclude_tol_deg: float = 0.6,
                              vote_theta_range: Tuple[float, float] = (0.0, 180.0),
                              save_candidate_overlays: bool = False
                              ) -> Dict[str, Any]:
    # 读取灰度帧
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frames: List[np.ndarray] = []
    count = 0
    probe_rows = []
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames > 0 and count >= max_frames):
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(g); count += 1
    cap.release()
    if len(frames) == 0:
        raise RuntimeError("读取到 0 帧")

    t_total0 = time.perf_counter()
    angle_times: List[Dict[str, float]] = []
    if vote_exclude_normals is None:
        vote_exclude_normals = [45.0, 135.0]

    best: Dict[str, Any] = {
        "angle": None, "slope": None, "score": -1.0,
        "theta_fft": None, "sti_raw": None, "fps": fps, "angle_probe": None
    }

    n_lines = 0
    a = angle_start    #测速线角度
    while a <= angle_end + 1e-6:
        t0 = time.perf_counter(); n_lines += 1
        sti = build_sti_from_frames(frames, center, length_px, angle_deg=a)
        if sti is None:
            a += angle_step; continue

        sti_in = sti; theta_fft = None
        if use_fft_fan_filter:
            sti_in, theta_fft = enhance_sti_via_fft_fan(
                sti, half_width_deg=fft_half_width_deg,
                rmin_ratio=fft_rmin_ratio, rmax_ratio=fft_rmax_ratio
            )

        edges = compute_canny_edges(
            sti_in, use_circular_roi=use_circular_roi,
            save_name="step7_canny_edges_tmp.png", verbose=False
        )

        # —— 6 项解包（论文口径 + 双线性入桶）——
        total, angle_votes, votes_full, theta_axis, rho_max, best_info = hough_angle_voting_min(
            edges,
            theta_res_deg=vote_theta_res_deg,
            rho_step=vote_rho_step,  # ← 用你真正传给 Hough 的 rho_step
            k_ratio=float(vote_k_ratio),
            verbose=False
        )
        rho_bins = int(np.floor((2 * rho_max) / vote_rho_step) + 1)

        # 角度过滤
        votes_filtered = _apply_theta_filters_on_votes(
            votes_full, theta_axis,
            exclude_normals=vote_exclude_normals,
            exclude_tol_deg=vote_exclude_tol_deg,
            theta_range=vote_theta_range
        )
        if votes_filtered.sum() <= 0:
            #
            # 记录一行（无峰时，得分=0）
            H, W = edges.shape[:2]
            r = min(W / 2.0, H / 2.0)
            K_here = int(max(1, round(float(vote_k_ratio) * r)))
            rho_bins = int(np.floor((2 * rho_max) / 1.0) + 1)
            probe_rows.append({
                "probe_angle_deg": float(a),
                "phi_star_deg": float("nan"),
                "alpha_star_deg": float("nan"),
                "score_lines": int(0),
                "rho_max": int(rho_max),
                "rho_bins": int(rho_bins),
                "K": int(K_here),
            })
            #
            angle_times.append({"angle": float(a), "seconds": float(time.perf_counter() - t0)})
            a += angle_step;continue

        peak_idx = int(np.argmax(votes_filtered))
        theta_normal_deg = float(theta_axis[peak_idx])
        peak_votes = float(votes_filtered[peak_idx])        # = 该 θ 上“≥K 的 ρ-bin 个数”
        alpha_deg = (theta_normal_deg + 90.0) % 180.0
        tan_a = math.tan(math.radians(alpha_deg))
        slope = None if abs(tan_a) < 1e-9 else (1.0 / tan_a)

        #
        # —— 记录/打印本角度的 ρ 参数与得分 —— #
        H, W = edges.shape[:2]
        r = min(W / 2.0, H / 2.0)
        K_here = int(max(1, round(float(vote_k_ratio) * r)))
        rho_bins = int(np.floor((2 * rho_max) / 1.0) + 1)

        probe_rows.append({
            "probe_angle_deg": float(a),
            "phi_star_deg": float(theta_normal_deg),
            "alpha_star_deg": float(alpha_deg),
            "score_lines": int(peak_votes),
            "rho_max": int(rho_max),
            "rho_bins": int(rho_bins),
            "K": int(K_here),
        })

        # 也在控制台打一行，便于你现场看
        print(f"[angle] a={a:+06.1f}° | score={int(peak_votes)} | φ*={theta_normal_deg:.1f}° | "
              f"ρ_max={int(rho_max)} | ρ_bins={int(rho_bins)} | K={int(K_here)}")
       #

        if save_candidate_overlays:
            _draw_line_overlay(
                sti, alpha_deg=alpha_deg, theta_normal_deg=theta_normal_deg,
                slope=slope, peak_votes=peak_votes,
                save_name=f"step8_hough_overlay_{a:+06.1f}.png"
            )

        angle_times.append({"angle": float(a), "seconds": float(time.perf_counter() - t0)})

        if peak_votes > best["score"]:
            best.update(dict(
                angle=alpha_deg, slope=slope, score=peak_votes,
                theta_fft=theta_fft, sti_raw=sti, angle_probe=a
            ))
        a += angle_step

    # —— 用最佳角度再落盘一次 —— #
    if best["angle"] is not None:
        a_best = best.get("angle_probe", angle_start)
        sti_best = build_sti_from_frames(frames, center, length_px, angle_deg=a_best)
        best["sti_raw"] = sti_best
        if use_fft_fan_filter and sti_best is not None:
            _ = enhance_sti_via_fft_fan(
                sti_best, half_width_deg=fft_half_width_deg,
                rmin_ratio=fft_rmin_ratio, rmax_ratio=fft_rmax_ratio
            )
        edges_best = compute_canny_edges(
            sti_best, use_circular_roi=use_circular_roi,
            save_name="step7_canny_edges.png", verbose=verbose
        )

        total, angle_votes, votes_full, theta_axis, _, _ = hough_angle_voting_min(
            edges_best, theta_res_deg=vote_theta_res_deg, rho_step=1.0, k_ratio=float(vote_k_ratio)
        )
        votes_filtered = _apply_theta_filters_on_votes(
            votes_full, theta_axis,
            exclude_normals=vote_exclude_normals,
            exclude_tol_deg=vote_exclude_tol_deg,
            theta_range=vote_theta_range
        )

        if votes_filtered.sum() > 0:
            peak_idx = int(np.argmax(votes_filtered))
            theta_normal_deg = float(theta_axis[peak_idx])
            peak_votes = float(votes_filtered[peak_idx])
            alpha_deg = (theta_normal_deg + 90.0) % 180.0
            tan_a = math.tan(math.radians(alpha_deg))
            slope = None if abs(tan_a) < 1e-9 else (1.0 / tan_a)

            best["angle"] = alpha_deg
            best["slope"] = slope
            best["score"] = peak_votes

            # 叠加图像（本地函数，不再依赖 core 导入）
            _draw_line_overlay(sti_best, alpha_deg=alpha_deg, theta_normal_deg=theta_normal_deg,
                               slope=slope, peak_votes=peak_votes, save_name="step8_hough_overlay.png")

    best["angle_times"] = angle_times
    best["num_lines"] = n_lines
    best["total_time_sec"] = float(time.perf_counter() - t_total0)

    if verbose:
        print(f"[search] 最优线方向角 α = {best['angle']}, 得分(主峰票数)={best['score']:.1f}, slope(px/frame)={best['slope']}")
        print(f"[search] 测速线数量={n_lines}, 总用时={best['total_time_sec']:.3f}s")


    #
    # —— 将本轮扫描的“每角结果”输出为 CSV —— #
    csv_path = "angle_scores.csv"
    try:
        # 优先写到调试目录（若存在）
        from .core import DEBUG_RUN_DIR
        if DEBUG_RUN_DIR:
            csv_path = os.path.join(DEBUG_RUN_DIR, "angle_scores.csv")
    except Exception:
        pass

    # 写 CSV
    try:
        fieldnames = ["probe_angle_deg", "phi_star_deg", "alpha_star_deg",
                      "score_lines", "rho_max", "rho_bins", "K"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(probe_rows, key=lambda d: d["probe_angle_deg"]):
                writer.writerow(row)
        if verbose:
            print(f"[angles.csv] 已保存每角结果: {csv_path}")
    except Exception as e:
        if verbose:
            print(f"[angles.csv] 保存失败: {e}")

    # 也打印一个简短汇总（前若干项）
    if verbose and len(probe_rows) > 0:
        top = sorted(probe_rows, key=lambda d: (-d["score_lines"], d["probe_angle_deg"]))[:10]
        print("[angles] Top-10 by score_lines:")
        for r0 in top:
            print(f"  a={r0['probe_angle_deg']:+06.1f}° | score={r0['score_lines']:3d} | "
                  f"φ*={r0['phi_star_deg']:6.1f}° | ρ_max={r0['rho_max']:3d} | "
                  f"ρ_bins={r0['rho_bins']:3d} | K={r0['K']}")

    #
    return best


# def calculate_extended_line(center: Tuple[int, int], bank_point: Tuple[int, int], interval_px: int) -> List[
#     Tuple[int, int]]:
#     # 计算两点之间的向量
#     cx, cy = center
#     bx, by = bank_point
#     vector_x = bx - cx
#     vector_y = by - cy
#     length = int(np.sqrt(vector_x ** 2 + vector_y ** 2))  # 计算两点间的距离
#     num_points = length // interval_px  # 计算测点数目
#     extended_points = []
#
#     for i in range(-num_points, num_points + 1):
#         # 计算每个测点坐标
#         new_x = int(cx + i * vector_x / num_points)
#         new_y = int(cy + i * vector_y / num_points)
#         extended_points.append((new_x, new_y))
#
#     return extended_points
#
#
# # 批量测速：沿线段计算多点测速结果
# def batch_probe_along_line(video_path: str,
#                            center: Tuple[int, int],
#                            bank_point: Tuple[int, int],
#                            interval_px: int,
#                            length_px: int,
#                            angle_range: Tuple[float, float, float],
#                            max_frames: int,
#                            m_per_px: float,
#                            fps: float,
#                            use_circular_roi: bool,
#                            use_fft_fan_filter: bool,
#                            fft_half_width_deg: float,
#                            fft_rmin_ratio: float,
#                            fft_rmax_ratio: float,
#                            vote_theta_res_deg: float,
#                            vote_k_ratio: float,
#                            vote_exclude_normals: List[float],
#                            vote_exclude_tol_deg: float,
#                            vote_theta_range: Tuple[float, float]) -> None:
#     # 初始化视频读取
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError(f'无法打开视频文件: {video_path}')
#     fps = float(cap.get(cv2.CAP_PROP_FPS))
#
#     # 获取所有视频帧
#     frames = []
#     count = 0
#     while True:
#         ok, frame = cap.read()
#         if not ok or (max_frames > 0 and count >= max_frames):
#             break
#         frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
#         count += 1
#     cap.release()
#
#     # 计算所有延伸点
#     extended_points = calculate_extended_line(center, bank_point, interval_px)
#
#     # 存储每个点的测速结果
#     results = []
#
#     # 对每个测点执行相同的测速流程
#     for point in extended_points:
#         sti = build_sti_from_frames(frames, point, length_px, angle_range[0])  # 初步计算 STI
#         if sti is None:
#             continue
#
#         # 可选 FFT 扇形增强
#         if use_fft_fan_filter:
#             sti, _ = enhance_sti_via_fft_fan(sti, fft_half_width_deg, fft_rmin_ratio, fft_rmax_ratio)
#
#         # Canny 边缘提取
#         edges = compute_canny_edges(sti, use_circular_roi)
#
#         # 角度投票霍夫
#         total_lines, angle_votes, votes_full, theta_axis, rho_max, best_info = hough_angle_voting_min(
#             edges, theta_res_deg=vote_theta_res_deg, rho_step=1.0, k_ratio=vote_k_ratio)
#
#         # 过滤掉不符合条件的角度
#         votes_filtered = _apply_theta_filters_on_votes(votes_full, theta_axis, vote_exclude_normals,
#                                                        vote_exclude_tol_deg, vote_theta_range)
#
#         # 获取最佳结果
#         if np.sum(votes_filtered) > 0:
#             peak_idx = np.argmax(votes_filtered)
#             theta_normal_deg = theta_axis[peak_idx]
#             peak_votes = votes_filtered[peak_idx]
#             alpha_deg = (theta_normal_deg + 90) % 180  # 计算主纹理角
#             results.append({
#                 'point': point,
#                 'theta_normal_deg': theta_normal_deg,
#                 'alpha_deg': alpha_deg,
#                 'speed_m_per_s': peak_votes * m_per_px * fps,
#                 'hough_score': peak_votes
#             })
#         else:
#             results.append({
#                 'point': point,
#                 'theta_normal_deg': None,
#                 'alpha_deg': None,
#                 'speed_m_per_s': None,
#                 'hough_score': None
#             })
#
#     # 保存结果为 Excel
#     df = pd.DataFrame(results)
#     df.to_excel('batch_probe_results.xlsx', index=False)
#     print(f"[result] 批量测速结果已保存到 'batch_probe_results.xlsx'")