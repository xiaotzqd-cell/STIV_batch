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
    build_sti_from_frames,
    enhance_sti_via_fft_fan,
    compute_canny_edges,
    push_debug_dir,
    DEBUG_RUN_DIR,
)
from .vote_accumulator import hough_angle_voting_min

vote_rho_step = 1

def _apply_theta_filters_on_votes(votes_full: np.ndarray,
                                  theta_axis: np.ndarray,
                                  theta_range: Tuple[float, float]) -> np.ndarray:
    """根据角度范围对 votes_full 进行裁剪。"""
    vf = votes_full.copy()
    th_min, th_max = theta_range
    valid = (theta_axis >= th_min) & (theta_axis < th_max)
    vf[~valid] = 0
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


def _load_video_frames(video_path: str, max_frames: int) -> Tuple[List[np.ndarray], float]:
    """读取视频帧并返回灰度帧列表及 FPS。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not fps or math.isinf(fps) or math.isnan(fps):
        fps = 30.0
    frames: List[np.ndarray] = []
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames > 0 and count >= max_frames):
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        count += 1
    cap.release()
    if not frames:
        raise RuntimeError("读取到 0 帧")
    return frames, fps


def _adaptive_direction_search_on_frames(
    frames: List[np.ndarray],
    fps: float,
    center: Tuple[int, int],
    length_px: int,
    angle_start: float,
    angle_end: float,
    angle_step: float,
    *,
    use_circular_roi: bool,
    use_fft_fan_filter: bool,
    fft_half_width_deg: float,
    fft_rmin_ratio: float,
    fft_rmax_ratio: float,
    verbose: bool,
    vote_theta_res_deg: float,
    vote_k_ratio: float,
    vote_theta_range: Tuple[float, float],
    save_candidate_overlays: bool,
) -> Dict[str, Any]:
    probe_rows: List[Dict[str, Any]] = []
    t_total0 = time.perf_counter()
    angle_times: List[Dict[str, float]] = []

    best: Dict[str, Any] = {
        "angle": None,
        "slope": None,
        "score": -1.0,
        "theta_fft": None,
        "sti_raw": None,
        "fps": fps,
        "angle_probe": None,
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
            votes_full,
            theta_axis,
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
                              vote_theta_range: Tuple[float, float] = (0.0, 180.0),
                              save_candidate_overlays: bool = False
                              ) -> Dict[str, Any]:
    frames, fps = _load_video_frames(video_path, max_frames)

    return _adaptive_direction_search_on_frames(
        frames,
        fps,
        center,
        length_px,
        angle_start,
        angle_end,
        angle_step,
        use_circular_roi=use_circular_roi,
        use_fft_fan_filter=use_fft_fan_filter,
        fft_half_width_deg=fft_half_width_deg,
        fft_rmin_ratio=fft_rmin_ratio,
        fft_rmax_ratio=fft_rmax_ratio,
        verbose=verbose,
        vote_theta_res_deg=vote_theta_res_deg,
        vote_k_ratio=vote_k_ratio,
        vote_theta_range=vote_theta_range,
        save_candidate_overlays=save_candidate_overlays,
    )


def _calculate_extended_line(center: Tuple[int, int],
                              bank_point: Tuple[int, int],
                              interval_px: int,
                              frame_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """沿着 CENTER-岸边线生成多点测速坐标，仅覆盖 bank_point 与对岸对称点之间的区段。"""
    if interval_px <= 0:
        raise ValueError("interval_px 必须为正数")

    h, w = frame_shape
    cx, cy = center
    bx, by = bank_point
    dx = bx - cx
    dy = by - cy
    half_length = math.hypot(dx, dy)
    if half_length == 0:
        return [center]

    ux = dx / half_length
    uy = dy / half_length
    # 岸边点关于中心点的对称点，定义测速范围的另一端
    another_bank_point = (2 * cx - bx, 2 * cy - by)

    points: List[Tuple[int, int]] = [center]
    for direction in (1, -1):
        dist = interval_px
        # 仅在 bank_point 与其对岸对称点之间取样
        while dist <= half_length + 1e-6:
            px = cx + direction * ux * dist
            py = cy + direction * uy * dist
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            pt = (int(round(px)), int(round(py)))
            if pt not in points:
                points.append(pt)
            dist += interval_px

    # 确保两端点被纳入（若落在画面内）
    for endpoint in (bank_point, another_bank_point):
        ex, ey = endpoint
        if 0 <= ex < w and 0 <= ey < h and endpoint not in points:
            points.append(endpoint)

    return points


def batch_probe_along_line(
    video_path: str,
    center: Tuple[int, int],
    bank_point: Tuple[int, int],
    interval_px: int,
    length_px: int,
    angle_range: Tuple[float, float, float],
    max_frames: int,
    m_per_px: Optional[float],
    fps: Optional[float],
    use_circular_roi: bool,
    use_fft_fan_filter: bool,
    fft_half_width_deg: float,
    fft_rmin_ratio: float,
    fft_rmax_ratio: float,
    vote_theta_res_deg: float,
    vote_k_ratio: float,
    vote_theta_range: Tuple[float, float],
    verbose: bool,
) -> List[Dict[str, Any]]:
    """沿着给定直线执行多点测速。"""

    frames, video_fps = _load_video_frames(video_path, max_frames)
    effective_fps = fps if fps is not None else video_fps
    angle_start, angle_end, angle_step = angle_range

    frame_shape = frames[0].shape[:2]
    probe_points = _calculate_extended_line(center, bank_point, interval_px, frame_shape)

    results: List[Dict[str, Any]] = []
    excel_path = "batch_probe_results.xlsx"
    try:
        from .core import DEBUG_RUN_DIR
        if DEBUG_RUN_DIR:
            excel_path = os.path.join(DEBUG_RUN_DIR, excel_path)
    except Exception:
        pass

    for idx, point in enumerate(probe_points):
        best = _adaptive_direction_search_on_frames(
            frames,
            video_fps,
            point,
            length_px,
            angle_start,
            angle_end,
            angle_step,
            use_circular_roi=use_circular_roi,
            use_fft_fan_filter=use_fft_fan_filter,
            fft_half_width_deg=fft_half_width_deg,
            fft_rmin_ratio=fft_rmin_ratio,
            fft_rmax_ratio=fft_rmax_ratio,
            verbose=verbose,
            vote_theta_res_deg=vote_theta_res_deg,
            vote_k_ratio=vote_k_ratio,
            vote_theta_range=vote_theta_range,
            save_candidate_overlays=False,
        )

        if fps is not None:
            best["fps"] = float(fps)
        elif effective_fps:
            best["fps"] = float(effective_fps)

        slope = best.get("slope")
        best_fps = best.get("fps")
        speed_m_per_s = None
        if slope is not None and m_per_px is not None and best_fps:
            speed_m_per_s = abs(slope) * m_per_px * float(best_fps)

        result_row = {
            "index": idx,
            "point_x": point[0],
            "point_y": point[1],
            "angle_probe_deg": best.get("angle_probe"),
            "alpha_deg": best.get("angle"),
            "slope_px_per_frame": best.get("slope"),
            "speed_m_per_s": speed_m_per_s,
            "length_px": length_px,
            "score": best.get("score"),
        }
        results.append(result_row)

        if verbose:
            speed_txt = "N/A" if speed_m_per_s is None else f"{speed_m_per_s:.4f}"
            print(f"[batch] point#{idx:02d} {point} | length={length_px}px | speed={speed_txt} m/s")

    try:
        df = pd.DataFrame(results)
        df.to_excel(excel_path, index=False)
        if verbose:
            print(f"[batch] 多点测速结果已保存: {excel_path}")
    except Exception as exc:
        if verbose:
            print(f"[batch] 保存 Excel 失败: {exc}")

    return results
