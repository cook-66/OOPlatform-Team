#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : draw_utils.py
# @Project   : ODPlatform
# @Function  : 极简检测框绘制器 — 替代缺位的 visualization 包
"""极简绘制工具(只用 OpenCV).

提供:
  - Detection    : 一条检测(框 + 置信度 + 标签)
  - DrawStyle    : 线宽/字号/字色/标签底色不透明度, 支持按帧尺寸自适应
  - SimpleVisualizer : 在图像副本上画框 + 标签, 原图不动

刻意保持极简: 只画矩形框和文字, 不做字体加载 / 中文渲染 / 花哨样式.
font_path 字段保留但不生效(OpenCV 默认字体不支持中文) —— 需要中文标签时
请换成 PIL 方案, 那是超出 D7 范围的增强.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DrawStyle:
    """绘制样式. 默认值按 720p 调过, from_image_size 会按实际尺寸缩放."""
    box_thickness: int = 2
    text_scale: float = 0.6
    text_color: tuple[int, int, int] = (255, 255, 255)
    text_bg_alpha: float = 0.6

    @classmethod
    def from_image_size(cls, height: int, width: int, **overrides: Any) -> "DrawStyle":
        """按帧尺寸自适应出基础样式, 再用 overrides 覆盖(只认本类字段)."""
        short = max(1, min(height, width))
        base = cls(
            box_thickness=max(1, round(short / 360)),
            text_scale=max(0.3, round(short / 900, 2)),
        )
        allowed = {f for f in cls.__dataclass_fields__}
        for key, value in overrides.items():
            if key in allowed:
                setattr(base, key, value)
        return base


@dataclass(frozen=True)
class Detection:
    """一条检测结果."""
    xyxy: tuple[float, float, float, float]
    confidence: float
    label: str


class SimpleVisualizer:
    """极简检测框绘制器(对齐 BeautifyVisualizer 的用法, 但只依赖 cv2)."""

    def __init__(
        self,
        labels: list[str] | None = None,
        label_mapping: dict[str, str] | None = None,
        color_mapping: dict[str, tuple[int, int, int]] | None = None,
        default_color: tuple[int, int, int] = (0, 255, 0),
        font_path: str | None = None,
    ) -> None:
        self.labels = labels or []
        self.label_mapping = label_mapping or {}
        self.color_mapping = color_mapping or {}
        self.default_color = default_color
        self.font_path = font_path          # 保留字段, OpenCV 默认字体不生效

    def color_for(self, label: str) -> tuple[int, int, int]:
        return self.color_mapping.get(label, self.default_color)

    def draw(
        self,
        image: np.ndarray,
        detections: list[Detection],
        *,
        style: DrawStyle | None = None,
        use_label_mapping: bool = True,
    ) -> np.ndarray:
        """在 image 的副本上画所有框和标签, 返回副本(原图不动)."""
        import cv2

        canvas = image.copy()
        st = style or DrawStyle.from_image_size(*image.shape[:2])
        h, w = canvas.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det.xyxy)
            x1, x2 = sorted((max(0, min(x1, w - 1)), max(0, min(x2, w - 1))))
            y1, y2 = sorted((max(0, min(y1, h - 1)), max(0, min(y2, h - 1))))
            color = self.color_for(det.label)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, st.box_thickness)

            shown = self.label_mapping.get(det.label, det.label) if use_label_mapping else det.label
            text = f"{shown} {det.confidence:.2f}"
            (tw, th), base = cv2.getTextSize(text, font, st.text_scale, 1)

            # 标签优先画在框上方; 顶到画面外就落到框内
            ty = y1 - 4
            if ty - th - base < 0:
                ty = min(h - 1, y1 + th + base + 4)

            bg = canvas.copy()
            cv2.rectangle(bg, (x1, ty - th - base), (x1 + tw + 4, ty + base), color, -1)
            cv2.addWeighted(bg, st.text_bg_alpha, canvas, 1 - st.text_bg_alpha, 0, canvas)
            cv2.putText(canvas, text, (x1 + 2, ty), font, st.text_scale,
                        st.text_color, 1, cv2.LINE_AA)

        return canvas
