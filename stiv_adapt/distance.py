# -*- coding: utf-8 -*-
"""
distance.py — 标定比例尺（m/px）
运行后弹出标定图，让你用鼠标点两岸，按 Enter 保存，Esc 退出。
返回：比例尺 (m/px)、两点原始坐标、像素距离
"""
import os
import cv2
import math
import json
import time
# ====== 配置区 ======
IMAGE_PATH = r"D:\Programs\Python\stiv1\CRR_calibration_image.jpg"
REAL_DISTANCE_M = 49.38    # 两岸真实距离 (米)
OUT_DIR = os.path.join(os.path.dirname(__file__), "out_calib")
MAX_WIN_W, MAX_WIN_H = 1600, 1000   # 窗口最大尺寸，过大会等比例缩放显示
# ====================

points = []  # 存放点击的点
scale_factor = 1
orig_img = None
disp_img = None

def fit_to_window(w, h, max_w, max_h):
    sx = max_w / float(w)
    sy = max_h / float(h)
    return min(1.0, sx, sy)

def to_orig(pt_disp):
    """把显示坐标映射回原始分辨率"""
    x = int(round(pt_disp[0] / scale_factor))
    y = int(round(pt_disp[1] / scale_factor))
    return (x, y)

def on_mouse(event, x, y, flags, param):
    global points, disp_img
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        points = []

def calibrate_scale(image_path, real_distance_m):
    """
    弹窗选点，返回 (scale_m_per_pixel, (p1, p2), pixel_dist)
    p1, p2 是原始分辨率的坐标
    """
    global orig_img, disp_img, scale_factor, points
    points = []

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")

    orig_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if orig_img is None:
        raise RuntimeError("无法读取图片")

    H0, W0 = orig_img.shape[:2]
    scale_factor = fit_to_window(W0, H0, MAX_WIN_W, MAX_WIN_H)
    if scale_factor < 1.0:
        disp_img = cv2.resize(orig_img, (int(W0*scale_factor), int(H0*scale_factor)),
                              interpolation=cv2.INTER_AREA)
    else:
        disp_img = orig_img.copy()

    win = "Select two bank points (Left=add, Right=reset, Enter=save, Esc=quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = disp_img.copy()
        if len(points) >= 1:
            cv2.circle(vis, points[0], 6, (0,165,255), -1, cv2.LINE_AA)
        if len(points) == 2:
            cv2.circle(vis, points[1], 6, (0,165,255), -1, cv2.LINE_AA)
            cv2.line(vis, points[0], points[1], (0,165,255), 3, cv2.LINE_AA)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q'), ord('Q')):
            cv2.destroyAllWindows()
            return None, (None, None), 0.0
        #计算像素距离和m/px
        if k in (13, 10) and len(points) == 2:  # Enter
            p1 = to_orig(points[0])
            p2 = to_orig(points[1])
            px_dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            mpp = real_distance_m / px_dist

            # 保存结果
            os.makedirs(OUT_DIR, exist_ok=True)
            annotated = orig_img.copy()
            cv2.circle(annotated, p1, 8, (0,165,255), -1, cv2.LINE_AA)
            cv2.circle(annotated, p2, 8, (0,165,255), -1, cv2.LINE_AA)
            cv2.line(annotated, p1, p2, (0,165,255), 4, cv2.LINE_AA)

            text = f"Pixel={px_dist:.2f}px  Real={real_distance_m:.2f}m  Scale={mpp:.6f} m/px"
            cv2.putText(annotated, text, (15,40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (30,30,30), 3, cv2.LINE_AA)
            cv2.putText(annotated, text, (15,40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255,255,255), 2, cv2.LINE_AA)

            stamp = time.strftime("%Y%m%d-%H%M%S")
            out_img = os.path.join(OUT_DIR, f"calib_{stamp}.png")
            out_json = os.path.join(OUT_DIR, f"calib_{stamp}.json")
            cv2.imwrite(out_img, annotated)
            with open(out_json,"w",encoding="utf-8") as f:
                json.dump({
                    "image": os.path.abspath(image_path),
                    "p1": p1, "p2": p2,
                    "pixel_distance": px_dist,
                    "real_distance_m": real_distance_m,
                    "scale_m_per_pixel": mpp
                }, f, ensure_ascii=False, indent=2)

            print(f"[OK] Pixel={px_dist:.2f}px, Real={real_distance_m:.2f}m, Scale={mpp:.6f} m/px")
            print(f"[coords] p1={p1}, p2={p2}")
            print(f"[save] 标注图: {out_img}")
            print(f"[save] JSON:   {out_json}")
            cv2.destroyAllWindows()
            return mpp, (p1, p2), px_dist

if __name__ == "__main__":
    mpp, (p1, p2), px = calibrate_scale(IMAGE_PATH, REAL_DISTANCE_M)
    if mpp:
        print(f"\n最终结果: 1 px = {mpp:.9f} m, p1={p1}, p2={p2}, pixel_dist={px:.2f}")
