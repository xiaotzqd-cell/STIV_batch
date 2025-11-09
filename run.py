# -*- coding: utf-8 -*-
import os
import math
import cv2
from typing import Optional, Tuple, List, Dict
from stiv_adapt.search import adaptive_direction_search
from stiv_adapt.core import init_debug_dir
# ========== 用户配置区（按需修改） ==========
VIDEO = r"D:\Programs\Python\stiv\stiv_adapt/CRR.MP4"
CENTER: Tuple[int, int] =(1870, 1117)  # ← 手动中心点（像素坐标）

#多点测速参数
USE_BATCH_LINE_PROBING = True # ← 开启多点测速
BANK_POINT: Tuple[int, int] = (623, 1040) # 岸边点（与 CENTER 组成测速直线）
PROBE_INTERVAL_PX = 100 # 两测点之间的像素间隔（从中心点向两端延伸）
# STI 测线参数（角度搜索范围：线方向）
LENGTH_PX = 200
USE_DYNAMIC_LINE_LENGTH = True  # ← 让测线长度随速度缩放
DYNAMIC_LENGTH_REFERENCE_SPEED = 1.0  # 速度=1.0 m/s 时使用 LENGTH_PX
DYNAMIC_LENGTH_MIN_PX = max(16, LENGTH_PX // 2)
DYNAMIC_LENGTH_MAX_PX = LENGTH_PX * 3
ANGLE_START, ANGLE_END, ANGLE_STEP = -120, -70, 1   # 遍历的“测速线角度”
MAX_FRAMES = 200
USE_ROI = True
VERBOSE = True

# 频域扇形增强（用于评分）
USE_FFT_FAN = True
FFT_HALF_DEG = 4
FFT_RMIN_RATIO = 0.15
FFT_RMAX_RATIO = 0.9
# 帧率（建议手动给准值；留 None 则使用视频元数据）
FPS: Optional[float] = 23.976
# 比例尺：二选一
SCALE_M_PER_PIXEL: Optional[float] = None  # A) 直接给（m/px）；不想手填则设 None 走 B)
CALIB_REAL_M: Optional[float] = 49.38      # B) 首帧两点标定（米）
CALIB_LINE_XYXY: Optional[Tuple[int, int, int, int]] = (445, 1321, 3080, 1439)
#投票霍夫的可调参数（法线角 θ 的设置）——
VOTE_THETA_RES_DEG = 1                 # 角度分辨率（度）
VOTE_K_RATIO: float = 0.52               # 用比例阈值 K=0.55*R
VOTE_EXCLUDE_NORMALS = [45.0, 135.0]     # 排除异常峰的法线角（度）
VOTE_EXCLUDE_TOL_DEG = 0                 # 容差（度），建议≈分辨率的一半
VOTE_THETA_RANGE = (0.0, 180.0)          # 有效法线角范围 [min, max)
# ==========================================
def compute_scale_from_first_frame(video_path: str,
                                   xyxy: Tuple[int, int, int, int],
                                   real_meters: float) -> float:
    """在视频首帧上用两点像素距离和真实距离求 m/px。"""
    cap = cv2.VideoCapture(video_path)
    ok, frame0 = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("无法读取视频首帧用于标定")
    x1, y1, x2, y2 = xyxy
    px = math.hypot(x2 - x1, y2 - y1)
    if px < 1:
        raise RuntimeError("标定两点太近或坐标不正确")
    m_per_px = real_meters / px
    print(f"[calib] 像素距离={px:.2f}px, 真实距离={real_meters:.3f}m -> SCALE_M_PER_PIXEL={m_per_px:.6f} m/px")
    return m_per_px


def _line_endpoints(center, length_px, angle_deg):
    cx, cy = center
    half = length_px / 2.0
    rad  = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    x1 = int(round(cx - half*dx)); y1 = int(round(cy - half*dy))
    x2 = int(round(cx + half*dx)); y2 = int(round(cy + half*dy))
    return (x1, y1, x2, y2), (dx, dy)


def save_flow_overlay(
    video_path: str,
    outdir: str,
    center: tuple,                    # (cx, cy)
    best_angle_deg: float,            # 你的最佳角度（线方向）
    length_px: int,                   # 测线像素长度
    slope_px_per_frame: float|None,   # dx/dy (px/frame)
    m_per_px: float|None,             # 比例尺，可为 None
    fps: float|None,                  # 帧率，可为 None
    calib_xyxy: tuple|None=None,      # (x1,y1,x2,y2)
    calib_real_m: float|None=None,    # 真实距离（米）
    filename: str="frame_overlay.png",
    preview_max_side: int=1280
):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("[overlay] 无法读取首帧"); return

    H, W = frame.shape[:2]

    # —— 可选：标定线（橙色） ——
    if calib_xyxy and calib_real_m:
        x1,y1,x2,y2 = calib_xyxy
        cv2.line(frame, (x1,y1), (x2,y2), (0,165,255), 3, cv2.LINE_AA)
        midx, midy = (x1+x2)//2, (y1+y2)//2
        cv2.putText(frame, f"Calib {calib_real_m:.2f} m",
                    (midx+10, midy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0,165,255), 2, cv2.LINE_AA)

    # —— 测速截面（黄色实线） ——
    (x1,y1,x2,y2), (dx,dy) = _line_endpoints(center, length_px, best_angle_deg)
    cv2.line(frame, (x1,y1), (x2,y2), (0,255,255), 4, cv2.LINE_AA)
    cv2.putText(frame, "Velocity Cross-section",
                (min(x1,x2)+10, min(y1,y2)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)

    # —— 流动方向箭头（绿色） ——
    sign = 1 if (slope_px_per_frame is None or slope_px_per_frame >= 0) else -1
    arrow_len = max(60, int(round(length_px * 0.15)))
    start = (int(center[0]), int(center[1]))
    end   = (int(center[0] + sign*dx*arrow_len), int(center[1] + sign*dy*arrow_len))
    cv2.arrowedLine(frame, start, end, (0,255,0), 4, tipLength=0.1)

    # —— 左上角信息：slope / m/px / FPS / v(m/s) ——
    def put(line, row):
        y = 35 + row*30
        cv2.putText(frame, line, (15,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20,20,20), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (15,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    put(f"center={center}, angle={best_angle_deg:.1f} deg, length={length_px}px", 0)
    put(f"slope = {('None' if slope_px_per_frame is None else f'{slope_px_per_frame:.6f}')} px/frame", 1)
    put(f"m/px = {('None' if m_per_px is None else f'{m_per_px:.6f}')}", 2)
    put(f"FPS  = {('None' if fps is None else f'{fps:.6f}')}", 3)

    if slope_px_per_frame is not None and m_per_px is not None and fps is not None:
        v_mps = abs(slope_px_per_frame) * m_per_px * fps
        put(f"v = {v_mps:.4f} m/s", 4)
    else:
        put("v = N/A (缺少 slope/mpp/FPS)", 4)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, filename)
    cv2.imwrite(out_path, frame)
    print(f"[overlay] {os.path.abspath(out_path)}")

    # 预览缩放
    max_side = max(H, W)
    if max_side > preview_max_side:
        scale = preview_max_side / float(max_side)
        prev = cv2.resize(frame, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
        prev_path = os.path.join(outdir, os.path.splitext(filename)[0] + "_preview.png")
        cv2.imwrite(prev_path, prev)
        print(f"[overlay] {os.path.abspath(prev_path)} (preview)")


def save_batch_overlays(
    video_path: str,
    outdir: str,
    center: Tuple[int, int],
    bank_point: Tuple[int, int],
    batch_results: List[Dict[str, object]],
    *,
    m_per_px: Optional[float],
    default_fps: Optional[float],
) -> None:
    """在首帧上绘制所有多点测速结果，并生成单点叠加图。"""

    if not batch_results:
        return

    cap = cv2.VideoCapture(video_path)
    ok, frame0 = cap.read()
    cap.release()
    if not ok:
        print("[batch overlay] 无法读取首帧，跳过叠加图保存")
        return

    overview = frame0.copy()
    cv2.line(overview, center, bank_point, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(overview, center, 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(overview, bank_point, 6, (0, 0, 255), -1, cv2.LINE_AA)

    colors = [
        (0, 255, 255),
        (0, 165, 255),
        (0, 255, 0),
        (255, 0, 255),
        (255, 0, 0),
        (255, 255, 0),
    ]

    overlay_dir = os.path.join(outdir, "batch_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    for row in batch_results:
        angle = row.get("angle_probe_deg")
        if angle is None:
            continue
        idx = int(row.get("index", 0))
        point = (int(row.get("point_x", 0)), int(row.get("point_y", 0)))
        length = int(row.get("length_px", LENGTH_PX))
        slope = row.get("slope_px_per_frame")
        fps_here = row.get("fps") or default_fps
        color = colors[idx % len(colors)]

        (x1, y1, x2, y2), _ = _line_endpoints(point, length, angle)
        cv2.line(overview, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        cv2.circle(overview, point, 4, color, -1, cv2.LINE_AA)

        text = f"#{idx:02d}"
        speed_val = row.get("speed_m_per_s")
        if speed_val is not None:
            text += f" {speed_val:.2f} m/s"
        cv2.putText(overview, text, (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        filename = f"batch_point_{idx:02d}_overlay.png"
        save_flow_overlay(
            video_path=video_path,
            outdir=overlay_dir,
            center=point,
            best_angle_deg=angle,
            length_px=length,
            slope_px_per_frame=slope,
            m_per_px=m_per_px,
            fps=fps_here,
            calib_xyxy=None,
            calib_real_m=None,
            filename=filename,
            preview_max_side=1280,
        )

    overview_path = os.path.join(overlay_dir, "batch_overview.png")
    cv2.imwrite(overview_path, overview)
    print(f"[batch overlay] {os.path.abspath(overview_path)}")

def main():
    if not os.path.isfile(VIDEO):
        raise FileNotFoundError(f"视频不存在: {VIDEO}")

    outdir = init_debug_dir(tag="stiv-accu-vote")
    print(f"[out] 所有步骤图将保存到：{outdir}")
    print(f"[cfg] CENTER={CENTER}, LENGTH_PX={LENGTH_PX}, ANGLES=({ANGLE_START},{ANGLE_END},{ANGLE_STEP}), "
          f"MAX_FRAMES={MAX_FRAMES}, USE_ROI={USE_ROI}")

    # 计算/确定比例尺
    m_per_px = SCALE_M_PER_PIXEL
    if m_per_px is None and (CALIB_REAL_M is not None and CALIB_LINE_XYXY is not None):
        m_per_px = compute_scale_from_first_frame(VIDEO, CALIB_LINE_XYXY, CALIB_REAL_M)
    if m_per_px is not None:
        print(f"[scale] 使用 SCALE_M_PER_PIXEL={m_per_px:.6f} m/px")
    else:
        print("[scale] 未提供比例尺；将仅输出像素单位的斜率，不计算 m/s")

    if USE_BATCH_LINE_PROBING:
        from stiv_adapt.search import batch_probe_along_line

        results = batch_probe_along_line(
            video_path=VIDEO,
            center=CENTER,
            bank_point=BANK_POINT,
            interval_px=PROBE_INTERVAL_PX,
            length_px=LENGTH_PX,
            angle_range=(ANGLE_START, ANGLE_END, ANGLE_STEP),
            max_frames=MAX_FRAMES,
            m_per_px=m_per_px,
            fps=FPS,
            use_circular_roi=USE_ROI,
            use_fft_fan_filter=USE_FFT_FAN,
            fft_half_width_deg=FFT_HALF_DEG,
            fft_rmin_ratio=FFT_RMIN_RATIO,
            fft_rmax_ratio=FFT_RMAX_RATIO,
            vote_theta_res_deg=VOTE_THETA_RES_DEG,
            vote_k_ratio=VOTE_K_RATIO,
            vote_exclude_normals=VOTE_EXCLUDE_NORMALS,
            vote_exclude_tol_deg=VOTE_EXCLUDE_TOL_DEG,
            vote_theta_range=VOTE_THETA_RANGE,
            use_dynamic_length=USE_DYNAMIC_LINE_LENGTH,
            length_speed_reference=DYNAMIC_LENGTH_REFERENCE_SPEED,
            min_length_px=DYNAMIC_LENGTH_MIN_PX,
            max_length_px=DYNAMIC_LENGTH_MAX_PX,
            verbose=VERBOSE,
        )

        print("\n====== 多点测速结果 ======")
        for row in results:
            speed_txt = "N/A" if row["speed_m_per_s"] is None else f"{row['speed_m_per_s']:.4f} m/s"
            print(
                f"#{row['index']:02d} pt=({row['point_x']},{row['point_y']}) "
                f"len={row['length_px']}px angle={row['angle_probe_deg']}° "
                f"slope={row['slope_px_per_frame']} px/frame speed={speed_txt} score={row['score']}"
            )
        return


    # 自适应方向搜索（内部：构建STI → 可选FFT扇形增强 → Canny → 角度投票霍夫）
    best = adaptive_direction_search(
        video_path=VIDEO,
        center=CENTER,
        length_px=LENGTH_PX,
        angle_start=ANGLE_START,
        angle_end=ANGLE_END,
        angle_step=ANGLE_STEP,
        max_frames=MAX_FRAMES,
        use_circular_roi=USE_ROI,
        use_fft_fan_filter=USE_FFT_FAN,
        fft_half_width_deg=FFT_HALF_DEG,
        fft_rmin_ratio=FFT_RMIN_RATIO,
        fft_rmax_ratio=FFT_RMAX_RATIO,
        verbose=VERBOSE,
        # —— 将 run 的可调参数传入 search —— #
        vote_theta_res_deg=VOTE_THETA_RES_DEG,
        vote_k_ratio=VOTE_K_RATIO,
        vote_exclude_normals=VOTE_EXCLUDE_NORMALS,
        vote_exclude_tol_deg=VOTE_EXCLUDE_TOL_DEG,
        vote_theta_range=VOTE_THETA_RANGE,
        #vote_rho_step=VOTE_RHO_STEP,
    )

    # 覆写/确认 FPS
    if FPS is not None:
        best["fps"] = float(FPS)
    if not best.get("fps"):
        print("[warn] 无法可靠获取 FPS；建议在配置区手动设置 FPS。")
    else:
        print(f"[fps] 视频 FPS={best['fps']:.6f}")

    # 叠加到首帧预览图
    slope = best["slope"]      # dx/dy (px/frame)
    save_flow_overlay(
        video_path=VIDEO,
        outdir=outdir,
        center=CENTER,
        best_angle_deg=best.get("angle_probe", best["angle"]),
        length_px=LENGTH_PX,
        slope_px_per_frame=slope,
        m_per_px=m_per_px,
        fps=best.get("fps"),
        calib_xyxy=CALIB_LINE_XYXY,
        calib_real_m=CALIB_REAL_M,
        filename="frame_overlay.png",
        preview_max_side=1280
    )

    # 打印结果与速度换算
    print("\n====== 最终结果 ======")
    print(f"中心点: {CENTER}")
    #print(f"最佳条纹角度: {best['angle']} °")

    print(f"测速线方向: {best.get('angle_probe'):} °")
    print(f"最佳纹理角度α: {best['angle']:} °")

    print(f"Hough 得分(交线频率): {best['score']:.1f}")
    print(f"STI 斜率 slope (px/frame): {best['slope'] if best['slope'] is not None else 'None'}")

    if m_per_px is not None and best["slope"] is not None and best.get("fps"):
        v_mps = best["slope"] * m_per_px * best["fps"]
        print(f"速度估计: {v_mps:.4f} m/s   (slope={best['slope']:.6f} px/frame, m/px={m_per_px:.6f}, FPS={best['fps']:.3f})")
    else:
        print("未计算速度：缺少 slope 或 m/px 或 FPS。")

    # 耗时统计
    n_lines = best.get("num_lines", 0)
    t_total = best.get("total_time_sec", 0.0)
    times = best.get("angle_times") or []

    print(f"测速线数量: {n_lines}")
    print(f"总用时: {t_total:.3f} s")

    if times:
        avg = sum(t["seconds"] for t in times) / len(times)
        slow = max(times, key=lambda t: t["seconds"])
        print(f"单条平均用时: {avg:.3f} s，最慢: {slow['angle']:.1f}° → {slow['seconds']:.3f} s")
        preview = ", ".join(f"{t['angle']:.1f}°:{t['seconds']:.3f}s" for t in times[:10])
        print(f"每条用时(前10): {preview}")

    print("所有步骤图已写入输出目录。")


if __name__ == "__main__":
    main()
