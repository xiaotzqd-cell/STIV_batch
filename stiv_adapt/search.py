# -*- coding: utf-8 -*-
"""
search.py — 自适应方向搜索（Canny + 角度投票霍夫）
"""
from typing import Tuple, Optional, Dict, Any, List
from typing import Iterable
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

        # 角度不过滤，直接使用原始票数
        votes_filtered = votes_full
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
        if verbose:
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
        votes_filtered = votes_full

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


# === 批量测速：以 center 为中点、与 bank_point 构成的对称中心线，按间隔生成测点并逐点测速 ===
from typing import Iterable

def _generate_centerline_points(center: Tuple[int,int], bank_point: Tuple[int,int], interval_px: int) -> List[Tuple[int,int]]:
    cx, cy = center
    bx, by = bank_point
    vx, vy = bx - cx, by - cy
    L = float(math.hypot(vx, vy))
    if L < 1e-6:
        return [center]
    ux, uy = vx / L, vy / L
    # 对称线段端点（长度 2L，center 为中点）
    # 按 interval 取样，包含 center，向两端扩展到不超过端点
    n = int(math.floor(L / max(1, interval_px)))
    pts = [(int(round(cx + k * interval_px * ux)),
            int(round(cy + k * interval_px * uy))) for k in range(-n, n+1)]
    # 始终保证 center 在列表中
    if center not in pts:
        pts.insert(n, center)
    return pts


def _draw_batch_overlay(first_frame_bgr: np.ndarray,
                        records: List[Dict[str, Any]],
                        arrow_len_px: int = 600,
                        save_name: str = "centerline_batch_overlay.png") -> str:
    """在首帧上叠加本次批量测速的点与箭头（失败点画 ×），返回保存路径"""
    vis = first_frame_bgr.copy()
    H, W = vis.shape[:2]

    # 底图淡化，方便标注更清楚（可选）
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.15, vis, 0.85, 0)

    for rec in records:
        x, y = int(rec["x"]), int(rec["y"])
        ok = bool(rec.get("success", False))
        # 画测点
        cv2.circle(vis, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
        if ok:
            # 成功点：实心绿点 + 箭头表示流向
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1, cv2.LINE_AA)
            # 箭头方向：沿线方向 alpha；用 slope 的符号决定箭头朝向
            alpha = float(rec["angle_line_deg"])
            dx, dy = math.cos(math.radians(alpha)), math.sin(math.radians(alpha))
            sign = 1 if (rec.get("slope_px_per_frame") is None or rec.get("slope_px_per_frame") >= 0) else -1
            end = (int(round(x + sign * dx * arrow_len_px)),
                   int(round(y + sign * dy * arrow_len_px)))
            cv2.arrowedLine(vis, (x, y), end, (0, 200, 255), 4, tipLength=0.15)
        else:
            # 失败点标 ×
            cv2.line(vis, (x-7, y-7), (x+7, y+7), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(vis, (x-7, y+7), (x+7, y-7), (0, 0, 255), 2, cv2.LINE_AA)

    # 保存到调试目录
    try:
        from .core import DEBUG_RUN_DIR, _ensure_dir
        outdir = DEBUG_RUN_DIR or init_debug_dir()
        _ensure_dir(outdir)
        out_path = os.path.join(outdir, save_name)
    except Exception:
        out_path = save_name
    cv2.imwrite(out_path, vis)
    return out_path


def batch_probe_along_centerline(
    video_path: str,
    center: Tuple[int, int],
    bank_point: Tuple[int, int],
    interval_px: int,
    length_px: int,
    angle_start: float, angle_end: float, angle_step: float,
    max_frames: int = 300,
    m_per_px: Optional[float] = None,
    fps: Optional[float] = None,
    use_circular_roi: bool = False,
    use_fft_fan_filter: bool = True,
    fft_half_width_deg: float = 4.0,
    fft_rmin_ratio: float = 0.05,
    fft_rmax_ratio: float = 1.0,
    vote_theta_res_deg: float = 0.5,
    vote_k_ratio: float = 0.55,
    vote_exclude_normals: Optional[List[float]] = None,
    vote_exclude_tol_deg: float = 0.6,
    vote_theta_range: Tuple[float, float] = (0.0, 180.0),
) -> Dict[str, Any]:
    """
    以 center 为中点、与 bank_point 构成长度为 2|center-bank| 的对称直线，
    按 interval_px 生成测点（包含 center），逐点执行与单点相同的测速流程（不在控制台逐点打印），
    并把结果写到 Excel，同时在首帧上做矢量叠加图。
    """
    # 读取首帧（叠加图用）
    cap = cv2.VideoCapture(video_path)
    ok, first = cap.read()
    cap.release()
    if not ok or first is None:
        raise RuntimeError("无法读取视频第一帧用于叠加图")

    # 生成测点列表
    pts = _generate_centerline_points(center, bank_point, interval_px)

    records: List[Dict[str, Any]] = []
    for i, p in enumerate(pts):
        # 逐点执行“单点测速流程”——与 run.py 中一致，仅关闭逐角打印
        res = adaptive_direction_search(
            video_path=video_path,
            center=p,
            length_px=length_px,
            angle_start=angle_start,
            angle_end=angle_end,
            angle_step=angle_step,
            max_frames=max_frames,
            use_circular_roi=use_circular_roi,
            use_fft_fan_filter=use_fft_fan_filter,
            fft_half_width_deg=fft_half_width_deg,
            fft_rmin_ratio=fft_rmin_ratio,
            fft_rmax_ratio=fft_rmax_ratio,
            verbose=False,
            vote_theta_res_deg=vote_theta_res_deg,
            vote_k_ratio=vote_k_ratio,
            vote_exclude_normals=vote_exclude_normals,
            vote_exclude_tol_deg=vote_exclude_tol_deg,
            vote_theta_range=vote_theta_range,
        )

        alpha_probe = res.get("angle_probe")  # 优先使用“测速线方向”
        alpha_texture = res.get("angle")      # 纹理方向 α（有时与测速线方向不同）
        theta_normal = res.get("theta_normal")  # 法线角（若提供，可用来推算）

        # 回退策略：若没有 angle_probe，就尝试由 theta_normal 推算；再不行才退到 angle
        if alpha_probe is None:
            if theta_normal is not None:
                alpha_probe = (float(theta_normal) + 90.0) % 180.0
            else:
                alpha_probe = alpha_texture

        slope = res.get("slope")
        theta_fft = res.get("theta_fft")
        score = res.get("score")
        ok = (alpha_probe is not None) and (slope is not None) and (score is not None) and (score > 0)

        # >>> 新增：在 append 之前计算 speed（m/s）
        speed = None
        if ok and (m_per_px is not None) and (fps is not None):
            try:
                speed = float(slope) * float(m_per_px) * float(fps)
            except Exception:
                speed = None
        # <<< 新增结束

        records.append(dict(
            point_idx=i,
            x=int(p[0]), y=int(p[1]),
            angle_line_deg=(float(alpha_probe) if alpha_probe is not None else float('nan')),
            speed_mps=(float(speed) if speed is not None else float('nan')),
            theta_fft_deg=(float(theta_fft) if theta_fft is not None else float('nan')),
            hough_score=(float(score) if score is not None else float('nan')),
            slope_px_per_frame=(float(slope) if slope is not None else float('nan')),
            success=bool(ok),
        ))


    # 写 Excel（若无 openpyxl 则退回 CSV）
    import pandas as pd
    df = pd.DataFrame.from_records(records, columns=[
        "point_idx", "x", "y", "angle_line_deg", "speed_mps",
        "theta_fft_deg", "hough_score", "slope_px_per_frame", "success"
    ])
    try:
        from .core import DEBUG_RUN_DIR, _ensure_dir
        outdir = DEBUG_RUN_DIR or init_debug_dir()
        _ensure_dir(outdir)
        xlsx_path = os.path.join(outdir, "centerline_batch_results.xlsx")
    except Exception:
        xlsx_path = "centerline_batch_results.xlsx"
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception:
        # 如果缺少 openpyxl，也写 CSV 兜底
        xlsx_path = os.path.splitext(xlsx_path)[0] + ".csv"
        df.to_csv(xlsx_path, index=False, encoding="utf-8")

    # 叠加首帧可视化
    overlay_path = _draw_batch_overlay(first, records, arrow_len_px=300)

    return {
        "points": pts,
        "records": records,
        "excel_path": xlsx_path,
        "overlay_path": overlay_path,
    }





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