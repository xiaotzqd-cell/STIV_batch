# -*- coding: utf-8 -*-
"""Sobel 边缘检测相关函数。"""
import cv2
import numpy as np
from typing import Optional


def _build_sobel_grad_mag(img: np.ndarray, use_highpass: bool = False, sigma: float = 9.0) -> np.ndarray:
    """计算 Sobel 梯度幅值，可选高斯高通预处理。"""
    # 如果开启高通，先做大尺度高斯模糊并与原图相减，突出高频细节
    if use_highpass:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        high = img.astype(np.float32) - blur.astype(np.float32)
        base = cv2.normalize(high, None, 0, 255, cv2.NORM_MINMAX)
    else:
        # 不做高通时，直接使用原图像素（float32）
        base = img.astype(np.float32)

    # Sobel 求 x/y 方向梯度
    gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
    # 梯度幅值，并归一化到 0~255
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def compute_sobel_edges(
    sti_u8: np.ndarray,
    use_highpass: bool = False,
    use_circular_roi: bool = False,
    roi_radius_frac: float = 1.0,
    save_mag_name: str = "step6_sobel_mag.png",
    save_edge_name: str = "step7_sobel_edges.png",
    verbose: bool = False,
    saver: Optional[callable] = None,
) -> np.ndarray:
    """基于 Sobel 的二值边缘提取。"""
    from .core import _circular_roi_mask, _save_img  # 延迟导入以避免循环引用

    H, W = sti_u8.shape[:2]
    sobel_mag = _build_sobel_grad_mag(sti_u8, use_highpass=use_highpass)

    # 如启用 ROI，则只保留圆形区域内的梯度
    if use_circular_roi:
        mask = _circular_roi_mask((H, W), radius_frac=roi_radius_frac)
        sobel_mag = cv2.bitwise_and(sobel_mag, sobel_mag, mask=mask.astype(np.uint8))

    # 保存梯度幅值与阈值化后的边缘图
    saver = saver or _save_img
    saver(save_mag_name, sobel_mag)

    thr, edges = cv2.threshold(sobel_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    saver(save_edge_name, edges)

    if verbose:
        hp_txt = "highpass" if use_highpass else "direct"
        roi_txt = "circle" if use_circular_roi else "none"
        print(f"[sobel] mode={hp_txt}, roi={roi_txt}, otsu_threshold={int(thr)}")

    return edges
