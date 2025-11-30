# -*- coding: utf-8 -*-
"""Sobel 边缘检测相关函数。"""
import cv2
import numpy as np
from typing import Optional


def _build_sobel_grad_mag(img: np.ndarray, use_highpass: bool = False, sigma: float = 9.0) -> np.ndarray:
    """按照用户提供的示例实现 Sobel 梯度幅值（可选高斯高通）。"""
    # 方法1：直接计算 |∇I|
    if not use_highpass:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 方法2：先大尺度高斯模糊得到背景，计算高频分量后再求梯度
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    high = img.astype(np.float32) - blur.astype(np.float32)
    high_norm = cv2.normalize(high, None, 0, 255, cv2.NORM_MINMAX)

    gx = cv2.Sobel(high_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(high_norm, cv2.CV_32F, 0, 1, ksize=3)
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
    """基于 Sobel 梯度幅值的边缘提取（不再使用 Otsu 阈值）。"""
    from .core import _circular_roi_mask, _save_img  # 延迟导入以避免循环引用

    H, W = sti_u8.shape[:2]
    sobel_mag = _build_sobel_grad_mag(sti_u8, use_highpass=use_highpass)

    # 如启用 ROI，则只保留圆形区域内的梯度
    if use_circular_roi:
        mask = _circular_roi_mask((H, W), radius_frac=roi_radius_frac)
        sobel_mag = cv2.bitwise_and(sobel_mag, sobel_mag, mask=mask.astype(np.uint8))

    # 保存梯度幅值图（直接作为边缘图输入霍夫，无需再阈值化）
    saver = saver or _save_img
    saver(save_mag_name, sobel_mag)

    # 直接返回梯度幅值图（已归一化为 uint8）
    if save_edge_name:
        saver(save_edge_name, sobel_mag)

    if verbose:
        hp_txt = "highpass" if use_highpass else "direct"
        roi_txt = "circle" if use_circular_roi else "none"
        print(f"[sobel] mode={hp_txt}, roi={roi_txt}, use_highpass={use_highpass}")

    return sobel_mag
