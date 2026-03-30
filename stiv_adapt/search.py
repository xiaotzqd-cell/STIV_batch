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
    compute_canny_edges,
    push_debug_dir,
    DEBUG_RUN_DIR,
    _save_img,
)
from .sobel import (
    compute_sobel_edges,
    hough_angle_voting_weighted,
    DEFAULT_WEIGHT_MIN,
    DEFAULT_WEIGHT_MAX,
)
from .vote_accumulator import hough_angle_voting_min
from .autocorrelation import AutoCorrConfig, compute_autocorr_and_concentration

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


def _compute_symmetry_score(scores: np.ndarray, best_idx: int) -> float:
    """依据峰值左右半峰范围的对称性计算 E_asym。"""

    if scores.ndim != 1 or scores.size == 0:
        return float("nan")
    if best_idx < 0 or best_idx >= scores.size:
        return float("nan")

    best_score = float(scores[best_idx])
    half_score = 0.5 * best_score

    left_idx = best_idx
    while left_idx > 0 and scores[left_idx] > half_score:
        left_idx -= 1

    right_idx = best_idx
    last = scores.size - 1
    while right_idx < last and scores[right_idx] > half_score:
        right_idx += 1

    span_left = best_idx - left_idx
    span_right = right_idx - best_idx
    span = int(min(span_left, span_right))

    if span >= 1:
        num = 0.0
        den = 0.0
        for k in range(1, span + 1):
            s_left = float(scores[best_idx - k])
            s_right = float(scores[best_idx + k])
            num += abs(s_left - s_right)
            den += (s_left + s_right)
        return num / den if den > 0.0 else 0.0

    return float("nan")


def _compute_monotonicity_score(scores: np.ndarray, best_idx: int, peak_ratio: float) -> float:
    """依据峰值左右半峰范围的单调性计算 M_mono。

    逻辑：
    1) 以阈值 score_th = peak_ratio * peak 为界，分别找到左右半峰边界索引；
    2) 取 span=min(左跨度, 右跨度)，表示在半峰区间内可以向两侧走的步数；
    3) 从峰顶向左右各走 span 步，理想情况应当单调不增；
       若出现“当前值比前一值更大”，视为一次违例；
    4) N_step = 2*span，M_mono = 1 - N_violate/N_step。
    """

    if scores.ndim != 1 or scores.size == 0:
        return float("nan")
    if best_idx < 0 or best_idx >= scores.size:
        return float("nan")

    ratio = max(0.0, min(1.0, float(peak_ratio)))
    best_score = float(scores[best_idx])
    score_th = ratio * best_score

    left_idx = best_idx
    while left_idx > 0 and scores[left_idx] > score_th:
        left_idx -= 1

    right_idx = best_idx
    last = scores.size - 1
    while right_idx < last and scores[right_idx] > score_th:
        right_idx += 1

    span_left = best_idx - left_idx
    span_right = right_idx - best_idx
    span = int(min(span_left, span_right))

    if span < 1:
        return float("nan")

    N_step = 2 * span
    N_violate = 0
    eps = 0.0

    for k in range(1, span + 1):
        prev_val = float(scores[best_idx - (k - 1)])
        curr_val = float(scores[best_idx - k])
        if curr_val > prev_val + eps:
            N_violate += 1

    for k in range(1, span + 1):
        prev_val = float(scores[best_idx + (k - 1)])
        curr_val = float(scores[best_idx + k])
        if curr_val > prev_val + eps:
            N_violate += 1

    return 1.0 - (N_violate / float(N_step))


def _draw_line_overlay(sti_u8: np.ndarray,
                       alpha_deg: float,
                       theta_normal_deg: float,
                       slope: Optional[float],
                       peak_votes: float,
                       save_name: str) -> None:
    """在 STI 上叠加最佳线方向与文本说明，并保存到调试目录。"""
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

    # 根据图像尺寸动态调整文字大小，避免在高分辨率或低分辨率下被裁切
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = max(5, int(round(0.02 * min(H, W))))
    max_width = max(10, W - 2 * margin)
    font_scale = max(0.35, min(H, W) / 600.0)
    thickness = max(1, int(round(font_scale * 2)))
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    if text_w > max_width:
        scale_factor = max_width / float(text_w)
        font_scale = max(0.2, font_scale * scale_factor)
        thickness = max(1, int(round(font_scale * 2)))
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    text_org = (margin, min(H - baseline - 1, margin + text_h))
    cv2.rectangle(
        vis,
        (text_org[0] - 4, text_org[1] - text_h - baseline - 4),
        (text_org[0] + text_w + 4, text_org[1] + baseline + 4),
        (0, 0, 0),
        -1,
    )
    cv2.putText(vis, text, text_org, font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    # 将保存路径落到 DEBUG_RUN_DIR（若存在），确保文件与本次运行的其他输出在同一目录下
    path = save_name
    try:
        from .core import DEBUG_RUN_DIR, init_debug_dir  # 延迟导入以避免循环
        base = DEBUG_RUN_DIR or init_debug_dir()
        if base:
            path = os.path.join(base, save_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass

    cv2.imwrite(path, vis)


def _extract_peak_from_votes(
    votes_filtered: np.ndarray,
    theta_axis: np.ndarray,
    score_mode: str,
    *,
    sum_votes: Optional[float] = None,
) -> Dict[str, Any]:
    """根据角度投票结果提取峰值信息与线方向参数。"""
    if sum_votes is None:
        sum_votes = float(votes_filtered.sum())

    peak_idx = int(np.argmax(votes_filtered))
    theta_normal_deg = float(theta_axis[peak_idx])
    peak_votes = float(votes_filtered[peak_idx])
    peak_ratio = peak_votes / (sum_votes + 1e-9)
    alpha_deg = (theta_normal_deg + 90.0) % 180.0
    tan_a = math.tan(math.radians(alpha_deg))
    slope = None if abs(tan_a) < 1e-9 else (1.0 / tan_a)
    score_for_rank = peak_ratio if score_mode == "peak_ratio" else peak_votes

    return {
        "peak_idx": peak_idx,
        "theta_normal_deg": theta_normal_deg,
        "peak_votes": peak_votes,
        "peak_ratio": peak_ratio,
        "alpha_deg": alpha_deg,
        "slope": slope,
        "score_for_rank": score_for_rank,
        "sum_votes": float(sum_votes),
    }


def _load_video_frames(
    video_path: str,
    max_frames: int,
    start_frame: Optional[int] = None,
    start_time_sec: Optional[float] = None,
) -> Tuple[List[np.ndarray], float]:
    """读取视频帧并返回灰度帧列表及 FPS。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not fps or math.isinf(fps) or math.isnan(fps):
        fps = 30.0

    if start_frame is not None and start_time_sec is not None:
        cap.release()
        raise ValueError("start_frame 与 start_time_sec 只能设置一个")

    target_frame = 0
    if start_time_sec is not None:
        if start_time_sec < 0:
            cap.release()
            raise ValueError("start_time_sec 不能为负数")
        target_frame = int(round(float(start_time_sec) * fps))
    elif start_frame is not None:
        if start_frame < 0:
            cap.release()
            raise ValueError("start_frame 不能为负数")
        target_frame = int(start_frame)

    if target_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

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
    roi_radius_frac: float,
    edge_method: str,
    direction_method: str,
    verbose: bool,
    use_E_asym: bool,
    use_M_mono: bool,
    m_mono_peak_ratio: float,
    vote_theta_res_deg: float,
    vote_k_ratio: float,
    vote_theta_range: Tuple[float, float],
    k_sigma: float,
    save_candidate_overlays: bool,
    save_all_sti: bool,
    top_k_candidates: int,
    score_mode: str,
    save_debug_images: bool,
) -> Dict[str, Any]:
    """在给定帧序列上执行方向搜索并返回最佳结果。"""
    probe_rows: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    t_total0 = time.perf_counter()
    angle_times: List[Dict[str, float]] = []

    score_mode = (score_mode or "peak_votes").strip().lower()   #峰值和峰值占比选择
    if score_mode not in {"peak_votes", "peak_ratio"}:
        score_mode = "peak_votes"

    best: Dict[str, Any] = {
        "angle": None,
        "slope": None,
        "score": -1.0,
        "score_mode": score_mode,
        "peak_votes": None,
        "peak_ratio": None,
        "sti_raw": None,
        "sti_filtered": None,
        "fps": fps,
        "angle_probe": None,
        "E_asym": float("nan"),
        "M_mono": float("nan"),
    }

    edge_method = (edge_method or "canny").lower()
    if edge_method not in {"canny", "sobel"}:
        edge_method = "canny"

    direction_method = (direction_method or "hough").lower()
    if direction_method not in {"hough", "autocorr"}:
        direction_method = "hough"
    if direction_method == "autocorr":
        use_E_asym = False
        use_M_mono = False
        best["score_mode"] = "autocorr_c"

    n_lines = 0
    a = angle_start    #测速线角度
    while a <= angle_end + 1e-6:
        t0 = time.perf_counter(); n_lines += 1
        sti = build_sti_from_frames(frames, center, length_px, angle_deg=a)
        if sti is None:
            a += angle_step; continue

        if direction_method == "autocorr":
            if edge_method != "sobel":
                edge_method = "sobel"

            cfg = AutoCorrConfig(
                use_sobel_mag=True,
                use_circular_roi=use_circular_roi,
                roi_radius_frac=roi_radius_frac,
            )
            ac = compute_autocorr_and_concentration(sti, cfg)
            theta_deg = ac["theta_deg"]
            mu = ac["mu"]
            C = float(ac["C"])

            mu_max_idx = int(np.argmax(mu)) if mu.size > 0 else 0
            mu_max = float(mu[mu_max_idx]) if mu.size > 0 else float("nan")
            theta_deg_max = float(theta_deg[mu_max_idx]) if mu.size > 0 else float("nan")

            tan_a = math.tan(math.radians(a))
            slope = None if abs(tan_a) < 1e-9 else (1.0 / tan_a)
            score_for_rank = C

            row = {
                "probe_angle_deg": float(a),
                "C": float(C),
                "mu_max": float(mu_max),
                "theta_deg_max": float(theta_deg_max),
                "r_min": int(cfg.r_min),
                "r_max": int(cfg.r_max),
                "n_theta": int(cfg.n_theta),
                "baseline_mode": str(cfg.baseline_mode),
                "bottom_q": float(cfg.bottom_q),
                "use_sobel_mag": bool(cfg.use_sobel_mag),
                "use_circular_roi": bool(cfg.use_circular_roi),
                "roi_radius_frac": float(cfg.roi_radius_frac),
            }
            probe_rows.append(row)

            if verbose:
                print(f"[angle] a={a:+06.1f}° | C={C:.4f} | mu_max={mu_max:.4f} | theta_mu={theta_deg_max:.1f}°")

            angle_times.append({"angle": float(a), "seconds": float(time.perf_counter() - t0)})
            cand = {
                "angle": float(a),
                "slope": slope,
                "score": score_for_rank,
                "C": float(C),
                "mu_max": float(mu_max),
                "theta_deg_max": float(theta_deg_max),
                "angle_probe": a,
            }
            candidates.append(cand)

            if score_for_rank > best["score"]:
                best.update(dict(
                    angle=float(a), slope=slope, score=score_for_rank,
                    peak_votes=None,
                    peak_ratio=None,
                    sti_raw=sti, angle_probe=a,
                    E_asym=float("nan"), M_mono=float("nan"),
                ))
        else:
            sti_in = sti
            if save_debug_images:
                if save_all_sti:
                    _save_img(f"sti_raw/sti_raw_{a:+06.1f}.png", sti)
                else:
                    _save_img("step1_sti_raw.png", sti)

            if edge_method == "sobel":
                mag_name = None
                edge_name = None
                if save_debug_images:
                    mag_name = "step6_sobel_mag_tmp.png"
                    edge_name = "step7_sobel_edges_tmp.png"
                    if save_all_sti:
                        mag_name = f"sobel_mag/sobel_mag_{a:+06.1f}.png"
                        edge_name = f"sobel_edges/sobel_edges_{a:+06.1f}.png"
                edges = compute_sobel_edges(
                    sti_in,
                    use_circular_roi=use_circular_roi,
                    roi_radius_frac=roi_radius_frac,
                    save_mag_name=mag_name,
                    save_edge_name=edge_name,
                    verbose=False,
                )
                total, angle_votes, votes_full, theta_axis, rho_max, best_info = hough_angle_voting_weighted(
                    edges,
                    theta_res_deg=vote_theta_res_deg,
                    rho_step=vote_rho_step,
                    weight_min=DEFAULT_WEIGHT_MIN,
                    weight_max=DEFAULT_WEIGHT_MAX,
                    use_circular_roi=use_circular_roi,
                    roi_radius_frac=roi_radius_frac,
                    verbose=False,
                    k_sigma=k_sigma,
                )
            else:
                pre_canny_name = None
                canny_edge_name = None
                if save_debug_images:
                    pre_canny_name = "step6_pre_canny_eq_blur_tmp.png"
                    canny_edge_name = "step7_canny_edges_tmp.png"
                    if save_all_sti:
                        pre_canny_name = f"canny_pre/canny_pre_{a:+06.1f}.png"
                        canny_edge_name = f"canny_edges/canny_edges_{a:+06.1f}.png"
                edges = compute_canny_edges(
                    sti_in, use_circular_roi=use_circular_roi,
                    roi_radius_frac=roi_radius_frac,
                    save_name=canny_edge_name,
                    pre_canny_save_name=pre_canny_name,
                    verbose=False
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
            sum_votes = float(votes_filtered.sum())
            if sum_votes <= 0:
                #
                # 记录一行（无峰时，得分=0）
                H, W = edges.shape[:2]
                r = min(W / 2.0, H / 2.0)
                K_here = int(max(1, round(float(vote_k_ratio) * r)))
                rho_bins = int(np.floor((2 * rho_max) / 1.0) + 1)
                row = {
                    "probe_angle_deg": float(a),
                    "phi_star_deg": float("nan"),
                    "alpha_star_deg": float("nan"),
                    "score_lines": float(0.0),
                    "peak_ratio": float("nan"),
                    "rho_max": int(rho_max),
                    "rho_bins": int(rho_bins),
                    "K": int(K_here),
                }
                if use_E_asym:
                    row["E_asym"] = float("nan")
                if use_M_mono:
                    row["M_mono"] = float("nan")
                probe_rows.append(row)
                #
                angle_times.append({"angle": float(a), "seconds": float(time.perf_counter() - t0)})
                a += angle_step;continue

            peak_info = _extract_peak_from_votes(
                votes_filtered,
                theta_axis,
                score_mode,
                sum_votes=sum_votes,
            )
            peak_idx = peak_info["peak_idx"]
            theta_normal_deg = peak_info["theta_normal_deg"]
            peak_votes = peak_info["peak_votes"]
            peak_ratio = peak_info["peak_ratio"]
            alpha_deg = peak_info["alpha_deg"]
            slope = peak_info["slope"]
            score_for_rank = peak_info["score_for_rank"]

            E_asym = _compute_symmetry_score(votes_filtered, peak_idx) if use_E_asym else float("nan")
            M_mono = _compute_monotonicity_score(votes_filtered, peak_idx, m_mono_peak_ratio) if use_M_mono else float("nan")

            #
            # —— 记录/打印本角度的 ρ 参数与得分 —— #
            H, W = edges.shape[:2]
            r = min(W / 2.0, H / 2.0)
            K_here = int(max(1, round(float(vote_k_ratio) * r)))
            rho_bins = int(np.floor((2 * rho_max) / 1.0) + 1)

            row = {
                "probe_angle_deg": float(a),
                "phi_star_deg": float(theta_normal_deg),
                "alpha_star_deg": float(alpha_deg),
                "score_lines": float(peak_votes),
                "peak_ratio": float(peak_ratio),
                "rho_max": int(rho_max),
                "rho_bins": int(rho_bins),
                "K": int(K_here),
            }
            if use_E_asym:
                row["E_asym"] = float(E_asym)
            if use_M_mono:
                row["M_mono"] = float(M_mono)
            probe_rows.append(row)

            # 也在控制台打一行，便于你现场看
            score_txt = f"{peak_votes:.1f}" if edge_method == "sobel" else f"{int(round(peak_votes))}"
            ratio_txt = f" | ratio={peak_ratio:.4f}" if score_mode == "peak_ratio" else ""
            print(f"[angle] a={a:+06.1f}° | score={score_txt}{ratio_txt} | φ*={theta_normal_deg:.1f}° | "
                  f"ρ_max={int(rho_max)} | ρ_bins={int(rho_bins)} | K={int(K_here)}")


            if save_debug_images and save_candidate_overlays:
                _draw_line_overlay(
                    sti, alpha_deg=alpha_deg, theta_normal_deg=theta_normal_deg,
                    slope=slope, peak_votes=peak_votes,
                    save_name=f"step8_hough_overlay_{a:+06.1f}.png"
                )

            angle_times.append({"angle": float(a), "seconds": float(time.perf_counter() - t0)})

            cand = {
                "angle": alpha_deg,
                "slope": slope,
                "score": score_for_rank,
                "peak_votes": peak_votes,
                "peak_ratio": peak_ratio,
                "angle_probe": a,
            }
            if use_E_asym:
                cand["E_asym"] = float(E_asym)
            if use_M_mono:
                cand["M_mono"] = float(M_mono)
            candidates.append(cand)

            if score_for_rank > best["score"]:
                best.update(dict(
                    angle=alpha_deg, slope=slope, score=score_for_rank,
                    peak_votes=peak_votes,
                    peak_ratio=peak_ratio,
                    sti_raw=sti, angle_probe=a,
                    E_asym=E_asym, M_mono=M_mono,
                ))
        a += angle_step

    if candidates:
        k = top_k_candidates if top_k_candidates > 0 else len(candidates)
        top_candidates = sorted(candidates, key=lambda d: (-d["score"], d["angle_probe"]))[:k]
        if use_M_mono:
            def _mono_key(c: Dict[str, Any]):
                """为单调性优先策略生成排序键。"""
                mono = c.get("M_mono", float("nan"))
                mono_val = -math.inf if math.isnan(mono) else mono
                return (mono_val, c["score"], -c["angle_probe"])

            chosen = max(top_candidates, key=_mono_key)
        elif use_E_asym:
            def _asym_key(c: Dict[str, Any]):
                """为对称性优先策略生成排序键。"""
                asym = c.get("E_asym", float("nan"))
                return (math.inf if math.isnan(asym) else asym, -c["score"], c["angle_probe"])

            chosen = min(top_candidates, key=_asym_key)
        else:
            chosen = top_candidates[0]

        best.update(
            angle=chosen.get("angle"),
            slope=chosen.get("slope"),
            score=chosen.get("score", -1.0),
            peak_votes=chosen.get("peak_votes"),
            peak_ratio=chosen.get("peak_ratio"),
            angle_probe=chosen.get("angle_probe"),
            E_asym=chosen.get("E_asym", float("nan")),
            M_mono=chosen.get("M_mono", float("nan")),
        )

    # —— 用最佳角度再落盘一次 —— #
    if best["angle"] is not None and direction_method != "autocorr":
        a_best = best.get("angle_probe", angle_start)
        sti_best = build_sti_from_frames(frames, center, length_px, angle_deg=a_best)
        best["sti_raw"] = sti_best
        if edge_method == "sobel":
            edges_best = compute_sobel_edges(
                sti_best,
                use_circular_roi=use_circular_roi,
                roi_radius_frac=roi_radius_frac,
                save_mag_name="step6_sobel_mag.png",
                save_edge_name="step7_sobel_edges.png",
                verbose=verbose,
            )
            total, angle_votes, votes_full, theta_axis, _, _ = hough_angle_voting_weighted(
                edges_best,
                theta_res_deg=vote_theta_res_deg,
                rho_step=vote_rho_step,
                weight_min=DEFAULT_WEIGHT_MIN,
                weight_max=DEFAULT_WEIGHT_MAX,
                use_circular_roi=use_circular_roi,
                roi_radius_frac=roi_radius_frac,
                verbose=False,
                k_sigma=k_sigma,
            )
        else:
            edges_best = compute_canny_edges(
                sti_best,
                use_circular_roi=use_circular_roi,
                roi_radius_frac=roi_radius_frac,
                save_name="step7_canny_edges.png",
                pre_canny_save_name="step6_pre_canny_eq_blur.png",
                verbose=verbose,
            )

            total, angle_votes, votes_full, theta_axis, _, _ = hough_angle_voting_min(
                edges_best, theta_res_deg=vote_theta_res_deg, rho_step=vote_rho_step, k_ratio=float(vote_k_ratio)
            )
        votes_filtered = _apply_theta_filters_on_votes(
            votes_full,
            theta_axis,
            theta_range=vote_theta_range
        )

        sum_votes = float(votes_filtered.sum())
        if sum_votes > 0:
            peak_info = _extract_peak_from_votes(
                votes_filtered,
                theta_axis,
                score_mode,
                sum_votes=sum_votes,
            )
            peak_idx = peak_info["peak_idx"]
            theta_normal_deg = peak_info["theta_normal_deg"]
            peak_votes = peak_info["peak_votes"]
            peak_ratio = peak_info["peak_ratio"]
            alpha_deg = peak_info["alpha_deg"]
            slope = peak_info["slope"]
            score_for_rank = peak_info["score_for_rank"]

            E_asym = _compute_symmetry_score(votes_filtered, peak_idx) if use_E_asym else float("nan")
            M_mono = _compute_monotonicity_score(votes_filtered, peak_idx, m_mono_peak_ratio) if use_M_mono else float("nan")

            best["angle"] = alpha_deg
            best["slope"] = slope
            best["score"] = score_for_rank
            best["peak_votes"] = peak_votes
            best["peak_ratio"] = peak_ratio
            if use_E_asym:
                best["E_asym"] = E_asym
            if use_M_mono:
                best["M_mono"] = M_mono

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
        if direction_method == "autocorr":
            fieldnames = [
                "probe_angle_deg",
                "C",
                "mu_max",
                "theta_deg_max",
                "r_min",
                "r_max",
                "n_theta",
                "baseline_mode",
                "bottom_q",
                "use_sobel_mag",
                "use_circular_roi",
                "roi_radius_frac",
            ]
        else:
            fieldnames = ["probe_angle_deg", "phi_star_deg", "alpha_star_deg",
                          "score_lines", "peak_ratio", "rho_max", "rho_bins", "K"]
            if use_E_asym:
                fieldnames.append("E_asym")
            if use_M_mono:
                fieldnames.append("M_mono")
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
        top_num = top_k_candidates if top_k_candidates > 0 else len(probe_rows)
        if direction_method == "autocorr":
            top = sorted(probe_rows, key=lambda d: (-d["C"], d["probe_angle_deg"]))[:top_num]
            print(f"[angles] Top-{len(top)} by C:")
            for r0 in top:
                print(
                    f"  a={r0['probe_angle_deg']:+06.1f}° | C={r0['C']:.4f} | "
                    f"mu_max={r0['mu_max']:.4f} | theta_mu={r0['theta_deg_max']:.1f}°"
                )
        else:
            top = sorted(probe_rows, key=lambda d: (-d["score_lines"], d["probe_angle_deg"]))[:top_num]
            print(f"[angles] Top-{len(top)} by score_lines:")
            for r0 in top:
                score_val = r0["score_lines"]
                score_fmt = f"{score_val:.1f}" if edge_method == "sobel" else f"{int(round(score_val))}"
                print(
                    f"  a={r0['probe_angle_deg']:+06.1f}° | score={score_fmt:>7} | "
                    f"φ*={r0['phi_star_deg']:6.1f}° | ρ_max={r0['rho_max']:3d} | "
                    f"ρ_bins={r0['rho_bins']:3d} | K={r0['K']}"
                )

    #
    return best


def adaptive_direction_search(video_path: str,
                              center: Tuple[int, int],
                              length_px: int,
                              angle_start: float, angle_end: float, angle_step: float,
                              max_frames: int = 300,
                              start_frame: Optional[int] = None,
                              start_time_sec: Optional[float] = None,
                              use_circular_roi: bool = False,
                              roi_radius_frac: float = 1.0,
                              edge_method: str = "canny",
                              direction_method: str = "hough",
                              vote_theta_res_deg: float = 0.5,
                              vote_k_ratio: float = 0.55,
                              vote_theta_range: Tuple[float, float] = (0.0, 180.0),
                              verbose: bool = False,
                              *,
                              use_E_asym: bool = False,
                              use_M_mono: bool = False,
                              m_mono_peak_ratio: float = 0.5,
                              save_candidate_overlays: bool = False,
                              save_all_sti: bool = False,
                              top_k_candidates: int = 10,
                              k_sigma: float = 1.0,
                              score_mode: str = "peak_votes",
                              save_debug_images: bool = True,
                              ) -> Dict[str, Any]:
    """读取视频并执行自适应方向搜索。"""
    frames, fps = _load_video_frames(
        video_path,
        max_frames,
        start_frame=start_frame,
        start_time_sec=start_time_sec,
    )

    return _adaptive_direction_search_on_frames(
        frames,
        fps,
        center,
        length_px,
        angle_start,
        angle_end,
        angle_step,
        use_circular_roi=use_circular_roi,
        roi_radius_frac=roi_radius_frac,
        edge_method=edge_method,
        direction_method=direction_method,
        verbose=verbose,
        use_E_asym=use_E_asym,
        use_M_mono=use_M_mono,
        m_mono_peak_ratio=m_mono_peak_ratio,
        vote_theta_res_deg=vote_theta_res_deg,
        vote_k_ratio=vote_k_ratio,
        vote_theta_range=vote_theta_range,
        k_sigma=k_sigma,
        save_candidate_overlays=save_candidate_overlays,
        save_all_sti=save_all_sti,
        top_k_candidates=top_k_candidates,
        score_mode=score_mode,
        save_debug_images=save_debug_images,
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

    ux = dx / half_length  #方向单位化u=(ux,uy)
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
    # for endpoint in (bank_point, another_bank_point):
    #     ex, ey = endpoint
    #     if 0 <= ex < w and 0 <= ey < h and endpoint not in points:
    #         points.append(endpoint)

    return points


def batch_probe_along_line(
    video_path: str,
    center: Tuple[int, int],
    bank_point: Tuple[int, int],
    interval_px: int,
    length_px: int,
    angle_range: Tuple[float, float, float],
    max_frames: int,
    start_frame: Optional[int],
    start_time_sec: Optional[float],
    m_per_px: Optional[float],
    fps: Optional[float],
    use_circular_roi: bool,
    roi_radius_frac: float,
    edge_method: str,
    direction_method: str,
    vote_theta_res_deg: float,
    vote_k_ratio: float,
    vote_theta_range: Tuple[float, float],
    verbose: bool,
    *,
    coord_offset: Tuple[int, int] = (0, 0),
    use_E_asym: bool,
    use_M_mono: bool,
    m_mono_peak_ratio: float,
    top_k_candidates: int,
    k_sigma: float,
    score_mode: str,
    save_debug_images: bool,
) -> List[Dict[str, Any]]:
    """沿着给定直线执行多点测速。"""

    frames, video_fps = _load_video_frames(
        video_path,
        max_frames,
        start_frame=start_frame,
        start_time_sec=start_time_sec,
    )
    effective_fps = fps if fps is not None else video_fps
    angle_start, angle_end, angle_step = angle_range

    frame_shape = frames[0].shape[:2]
    probe_points_raw = _calculate_extended_line(center, bank_point, interval_px, frame_shape)

    # 以岸边点为首位，其余按与岸边点距离排序
    probe_points: List[Tuple[int, int]] = []
    seen = set()
    if bank_point in probe_points_raw:
        probe_points.append(bank_point)
        seen.add(bank_point)
    for pt in sorted(probe_points_raw, key=lambda p: math.hypot(p[0] - bank_point[0], p[1] - bank_point[1])):
        if pt not in seen:
            probe_points.append(pt)
            seen.add(pt)

    results: List[Dict[str, Any]] = []
    excel_path = "batch_probe_results.xlsx"
    try:
        from .core import DEBUG_RUN_DIR
        if DEBUG_RUN_DIR:
            excel_path = os.path.join(DEBUG_RUN_DIR, excel_path)
    except Exception:
        pass

    ox, oy = int(coord_offset[0]), int(coord_offset[1])

    for idx, point in enumerate(probe_points):
        point_global = (int(point[0] + ox), int(point[1] + oy))
        suffix = f"point_{idx:02d}_x{point_global[0]}_y{point_global[1]}"
        with push_debug_dir(suffix):
            best = _adaptive_direction_search_on_frames(
                frames,
                video_fps,
                point,
                length_px,
                angle_start,
                angle_end,
                angle_step,
                use_circular_roi=use_circular_roi,
                roi_radius_frac=roi_radius_frac,
                edge_method=edge_method,
                direction_method=direction_method,
                verbose=verbose,
                use_E_asym=use_E_asym,
                use_M_mono=use_M_mono,
                m_mono_peak_ratio=m_mono_peak_ratio,
                vote_theta_res_deg=vote_theta_res_deg,
                vote_k_ratio=vote_k_ratio,
                vote_theta_range=vote_theta_range,
                k_sigma=k_sigma,
                save_candidate_overlays=False,
                save_all_sti=False,
                top_k_candidates=top_k_candidates,
                score_mode=score_mode,
                save_debug_images=save_debug_images,
            )

            # 保存处理前后的 STI 到独立文件，便于逐点查看
            raw_sti = best.get("sti_raw")
            if raw_sti is not None:
                _save_img(f"{suffix}_sti_raw.png", raw_sti)
            filtered_sti = best.get("sti_filtered")
            if filtered_sti is not None:
                _save_img(f"{suffix}_sti_filtered.png", filtered_sti)

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
            "point_x": point_global[0],
            "point_y": point_global[1],
            "angle_probe_deg": best.get("angle_probe"),
            "alpha_deg": best.get("angle"),
            "slope_px_per_frame": best.get("slope"),
            "speed_m_per_s": speed_m_per_s,
            "length_px": length_px,
            "score": best.get("score"),
        }
        if use_E_asym:
            result_row["E_asym"] = best.get("E_asym")
        if use_M_mono:
            result_row["M_mono"] = best.get("M_mono")
        results.append(result_row)

        if verbose:
            speed_txt = "N/A" if speed_m_per_s is None else f"{speed_m_per_s:.4f}"
            mono_txt = "" if not use_M_mono else f" | M_mono={best.get('M_mono')}"
            print(f"[batch] point#{idx:02d} {point_global} | length={length_px}px | speed={speed_txt} m/s{mono_txt}")

    try:
        df = pd.DataFrame(results)
        df.to_excel(excel_path, index=False)
        if verbose:
            print(f"[batch] 多点测速结果已保存: {excel_path}")
    except Exception as exc:
        if verbose:
            print(f"[batch] 保存 Excel 失败: {exc}")

    return results
