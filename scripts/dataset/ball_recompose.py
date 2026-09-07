#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
不锈钢小球提取与轨道重组工具。

功能：
1. 在轨道 ROI 中使用 HoughCircles 检测小球；
2. 提取小球局部图、二值 mask 和带透明背景的 BGRA/RGBA 图；
3. 自动或手动限定轨道区域；
4. 使用 OpenCV inpaint 修复原始小球位置；
5. 将小球合成到多个轨道位置，或逐位置输出帧序列。

示例：
    python scripts/dataset/ball_recompose.py \
        --input ball_image/source/canmv-frame-2026-09-05T10-59-25-573Z.png \
        --output_dir ball_image/artifacts/output --mode multi

    python scripts/dataset/ball_recompose.py \
        --input test.png --output_dir ball_image/artifacts/output_sequence --mode sequence \
        --positions 200,350,500,650,800,950,1100

自动检测失败时，可以显式指定：
    python scripts/dataset/ball_recompose.py --input test.png --output_dir ball_image/artifacts/output \
        --ball 1220,382,26 --track-roi 0,320,1280,120

注意：OpenCV 内部通道顺序为 BGR/BGRA。保存的 PNG 带有 alpha 通道，
常见图像软件会把它显示为 RGBA 小球图。
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


Point = Tuple[int, int]
ROI = Tuple[int, int, int, int]
Position = Union[int, Point]


@dataclass(frozen=True)
class BallConfig:
    """检测和修复的默认参数，后续调参优先修改这里或使用命令行覆盖。"""

    # 小球尺寸约束
    min_radius: int = 16
    max_radius: int = 34
    expected_radius: float = 23.0

    # HoughCircles 参数
    hough_dp: float = 1.0
    hough_min_dist: float = 18.0
    hough_param1: float = 50.0
    hough_param2: float = 16.0
    gaussian_kernel: int = 9
    gaussian_sigma: float = 2.0

    # 反光球细化参数。Hough 有时会把球所在的棕色圆形凹槽当成大圆，
    # 因此在 Hough 结果附近再寻找“低饱和度 + 高亮度”的不锈钢区域。
    refine_reflection: bool = True
    reflection_sat_max: int = 100
    reflection_value_min: int = 90
    reflection_window_radius: int = 45
    reflection_min_area: int = 40

    # 已知画面中小球大致位于右侧；没有手动 ball-search-roi 时使用
    ball_search_x_ratio: float = 0.55
    ball_search_roi: Optional[ROI] = None
    expected_y_ratio: float = 0.53
    expected_y_tolerance: int = 42

    # 轨道颜色检测：棕色通常是低 H、高 S 的区域
    track_hue_min: int = 0
    track_hue_max: int = 25
    track_sat_min: int = 60
    track_value_min: int = 35
    track_value_max: int = 230
    track_profile_threshold: float = 0.45

    # 提取和背景修复
    patch_margin: int = 5
    feather_px: float = 2.0
    mask_dilate: int = 4
    inpaint_radius: float = 6.0


DEFAULT_CONFIG = BallConfig()


@dataclass(frozen=True)
class CircleInfo:
    """小球圆信息。center 是原图坐标，radius 单位为像素。"""

    center: Point
    radius: int


@dataclass(frozen=True)
class TrackInfo:
    """轨道区域和中心线信息。"""

    roi: ROI
    top_y: int
    bottom_y: int
    center_y: float
    x_start: int
    x_end: int
    slope: float = 0.0
    method: str = "auto"

    def center_y_at(self, x: float) -> int:
        """返回指定 x 处的轨道中心线 y。当前画面近似水平，保留 slope 便于后续扩展。"""

        y = self.center_y + self.slope * (float(x) - self.x_start)
        return int(round(np.clip(y, self.top_y, self.bottom_y)))


def _ensure_odd(value: int) -> int:
    """OpenCV 高斯核必须是正奇数。"""

    value = max(1, int(value))
    return value if value % 2 else value + 1


def _clip_roi(roi: ROI, width: int, height: int) -> ROI:
    """将 ROI 裁剪到图像范围内。"""

    x, y, w, h = map(int, roi)
    x1 = max(0, min(width, x))
    y1 = max(0, min(height, y))
    x2 = max(x1, min(width, x + max(0, w)))
    y2 = max(y1, min(height, y + max(0, h)))
    return x1, y1, x2 - x1, y2 - y1


def _parse_numbers(value: str, count: int, name: str) -> List[float]:
    """从 `x,y`、`x y` 或 `x,y,w,h` 文本中提取数字。"""

    numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
    if len(numbers) != count:
        raise argparse.ArgumentTypeError(
            f"{name} 需要 {count} 个数字，例如 "
            + ("x,y" if count == 2 else "x,y,w,h" if count == 4 else "x,y,r")
        )
    return [float(item) for item in numbers]


def parse_point(value: str) -> Point:
    numbers = _parse_numbers(value, 2, "点坐标")
    return int(round(numbers[0])), int(round(numbers[1]))


def parse_roi(value: str) -> ROI:
    numbers = _parse_numbers(value, 4, "ROI")
    return tuple(int(round(item)) for item in numbers)  # type: ignore[return-value]


def parse_circle(value: str) -> CircleInfo:
    numbers = _parse_numbers(value, 3, "小球参数")
    return CircleInfo(
        center=(int(round(numbers[0])), int(round(numbers[1]))),
        radius=max(1, int(round(numbers[2]))),
    )


def parse_positions(value: str) -> List[Position]:
    """解析 x 列表，也支持 `x1,y1;x2,y2` 形式的显式坐标。"""

    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("positions 不能为空")

    if ";" in value:
        result: List[Position] = []
        for item in value.split(";"):
            result.append(parse_point(item))
        return result

    numbers = re.findall(r"-?\d+", value)
    if not numbers:
        raise argparse.ArgumentTypeError("positions 需要逗号分隔的 x 坐标")
    return [int(item) for item in numbers]


def _unpack_circle(circle_info: Union[CircleInfo, Tuple[Point, int], Tuple[Tuple[int, int], int]]) -> CircleInfo:
    """兼容 CircleInfo 和用户要求的 ((cx, cy), radius) 形式。"""

    if isinstance(circle_info, CircleInfo):
        return circle_info

    center, radius = circle_info
    return CircleInfo(
        center=(int(center[0]), int(center[1])),
        radius=int(radius),
    )


def _draw_track_info(image: np.ndarray, track_info: TrackInfo) -> None:
    """在调试图上绘制轨道 ROI 和中心线。"""

    x, y, w, h = track_info.roi
    cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (255, 180, 0), 2)
    y_left = track_info.center_y_at(track_info.x_start)
    y_right = track_info.center_y_at(track_info.x_end)
    cv2.line(image, (track_info.x_start, y_left), (track_info.x_end, y_right), (0, 255, 255), 2)
    cv2.putText(
        image,
        f"track: {track_info.method}, center_y={track_info.center_y:.1f}",
        (max(5, x + 5), max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 180, 0),
        2,
        cv2.LINE_AA,
    )


def _track_color_mask(frame: np.ndarray, config: BallConfig) -> np.ndarray:
    """提取棕色轨道候选区域。该 mask 只用于寻找水平带状区域。"""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = (
        (hue >= config.track_hue_min)
        & (hue <= config.track_hue_max)
        & (saturation >= config.track_sat_min)
        & (value >= config.track_value_min)
        & (value <= config.track_value_max)
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def _find_active_runs(active: np.ndarray, offset: int, min_length: int = 8) -> List[Tuple[int, int]]:
    """从布尔行 profile 中提取连续区间。"""

    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None

    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            end = index
            if end - start >= min_length:
                runs.append((start + offset, end + offset))
            start = None

    if start is not None:
        end = len(active)
        if end - start >= min_length:
            runs.append((start + offset, end + offset))

    return runs


def detect_track_roi(
    frame: np.ndarray,
    manual_roi: Optional[ROI] = None,
    config: BallConfig = DEFAULT_CONFIG,
) -> TrackInfo:
    """
    检测或限定轨道 ROI。

    自动检测采用 HSV 棕色区域的逐行覆盖率，寻找画面中部最长的水平色带。
    对当前机械轨道画面，这比直接用全图边缘检测更稳定；如果颜色条件不适用，
    会回退到按画面比例生成的轨道 ROI。用户可以用 --track-roi 完全接管。
    """

    height, width = frame.shape[:2]

    if manual_roi is not None:
        roi = _clip_roi(manual_roi, width, height)
        x, y, w, h = roi
        if w < 8 or h < 8:
            raise ValueError(f"手动轨道 ROI 太小或越界: {manual_roi}")
        # ROI 是“轨道范围”而不一定正好落在槽中心，优先使用已知球心
        # y 比例；如果该位置不在手动 ROI 内，再退回到 ROI 几何中心。
        expected_y = height * config.expected_y_ratio
        center_y = float(np.clip(expected_y, y, y + h - 1))
        return TrackInfo(
            roi=roi,
            top_y=y,
            bottom_y=y + h - 1,
            center_y=center_y,
            x_start=x,
            x_end=x + w - 1,
            method="manual",
        )

    search_top = int(round(height * 0.25))
    search_bottom = int(round(height * 0.75))
    mask = _track_color_mask(frame, config)

    margin_x = max(0, int(round(width * 0.02)))
    row_profile = (mask[:, margin_x : max(margin_x + 1, width - margin_x)] > 0).mean(axis=1)
    profile_kernel = _ensure_odd(max(5, int(round(height * 0.02))))
    smooth_profile = cv2.GaussianBlur(
        row_profile.astype(np.float32).reshape(-1, 1),
        (1, profile_kernel),
        0,
    ).ravel()

    local_profile = smooth_profile[search_top:search_bottom]
    max_score = float(local_profile.max()) if local_profile.size else 0.0
    threshold = max(config.track_profile_threshold, max_score * 0.55)
    runs = _find_active_runs(smooth_profile[search_top:search_bottom] >= threshold, search_top)

    expected_y = int(round(height * config.expected_y_ratio))
    if runs:
        # 优先选择最长、覆盖率高且靠近已知小球 y 位置的连续色带。
        def run_score(run: Tuple[int, int]) -> float:
            top, bottom = run
            mean_score = float(smooth_profile[top:bottom].mean())
            length_score = min(1.0, (bottom - top) / max(1.0, height * 0.16))
            position_penalty = abs((top + bottom) / 2.0 - expected_y) / max(1.0, height * 0.35)
            return mean_score * 2.0 + length_score - position_penalty

        top_y, bottom_y = max(runs, key=run_score)
        pad = max(3, int(round(height * 0.008)))
        top_y = max(0, top_y - pad)
        bottom_y = min(height - 1, bottom_y + pad)
        method = "auto-color"
    else:
        # 当前输入图的经验回退范围约为 y=320..440。
        fallback_height = max(40, int(round(height * 0.17)))
        top_y = max(0, int(round(height * 0.44)))
        bottom_y = min(height - 1, top_y + fallback_height - 1)
        method = "fallback-ratio"

    if bottom_y <= top_y:
        top_y = max(0, int(round(height * 0.42)))
        bottom_y = min(height - 1, int(round(height * 0.62)))
        method = "fallback-ratio"

    # 已知球心 y 大约在 370..390；用该位置作为贴球中心线，
    # ROI 的 top/bottom 仍保留真实颜色带范围，方便检测约束和可视化。
    center_y = float(np.clip(expected_y, top_y, bottom_y))
    roi = (0, top_y, width, bottom_y - top_y + 1)
    return TrackInfo(
        roi=roi,
        top_y=top_y,
        bottom_y=bottom_y,
        center_y=center_y,
        x_start=0,
        x_end=width - 1,
        method=method if max_score >= 0.2 else "fallback-ratio",
    )


def _ring_edge_support(edges: np.ndarray, center: Point, radius: int) -> float:
    """计算圆周附近的边缘支持度，用于从多个 Hough 候选中选最佳圆。"""

    height, width = edges.shape[:2]
    cx, cy = center
    outer = np.zeros((height, width), dtype=np.uint8)
    inner = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(outer, (cx, cy), max(1, radius + 2), 255, -1)
    cv2.circle(inner, (cx, cy), max(1, radius - 2), 255, -1)
    ring = cv2.subtract(outer, inner) > 0
    if not np.any(ring):
        return 0.0
    return float(np.mean(edges[ring] > 0))


def _refine_reflective_circle(
    frame: np.ndarray,
    seed_center: Point,
    seed_radius: int,
    expected_y: float,
    config: BallConfig,
) -> Optional[CircleInfo]:
    """在 Hough 候选附近细化真正的不锈钢球轮廓。

    机械轨道中的球会产生白色高光，而球下面的棕色圆形凹槽也很容易
    形成一个完整的大圆。这里不完全依赖边缘，而是利用不锈钢区域通常
    “饱和度低、亮度高”的特点，在 Hough 圆附近做连通域筛选。

    该步骤只用于自动检测；手动传入 --ball 时会保留用户给出的圆参数。
    找不到可靠连通域时返回 None，让调用方继续使用 Hough 结果。
    """

    if not config.refine_reflection:
        return None

    height, width = frame.shape[:2]
    cx, cy = seed_center
    window_radius = max(
        20,
        int(config.reflection_window_radius),
        int(seed_radius) + 12,
    )

    x1 = max(0, cx - window_radius)
    y1 = max(0, cy - window_radius)
    x2 = min(width, cx + window_radius + 1)
    y2 = min(height, cy + window_radius + 1)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    reflective = (
        (saturation <= int(config.reflection_sat_max))
        & (value >= int(config.reflection_value_min))
    ).astype(np.uint8) * 255

    # 限制在候选圆附近，避免把右侧白色支架等亮物体选为小球。
    yy, xx = np.ogrid[: reflective.shape[0], : reflective.shape[1]]
    local_cx = cx - x1
    local_cy = cy - y1
    local_distance = (xx - local_cx) ** 2 + (yy - local_cy) ** 2
    reflective[local_distance > window_radius * window_radius] = 0

    # 去除零散噪点，但不做过强闭运算，避免把球和邻近的白色部件连接起来。
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    reflective = cv2.morphologyEx(reflective, cv2.MORPH_OPEN, open_kernel, iterations=1)

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        reflective,
        connectivity=8,
    )
    if component_count <= 1:
        return None

    best: Optional[Tuple[float, CircleInfo]] = None
    y_tolerance = max(float(config.expected_y_tolerance), float(seed_radius) + 12.0)
    max_component_size = max(60, int(config.max_radius * 2.5 + 12))

    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(config.reflection_min_area):
            continue

        box_x = int(stats[label, cv2.CC_STAT_LEFT])
        box_y = int(stats[label, cv2.CC_STAT_TOP])
        box_w = int(stats[label, cv2.CC_STAT_WIDTH])
        box_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if min(box_w, box_h) < 5 or max(box_w, box_h) > max_component_size:
            continue

        aspect = min(box_w, box_h) / max(1.0, float(max(box_w, box_h)))
        if aspect < 0.45:
            continue

        component_cx = x1 + float(centroids[label, 0])
        component_cy = y1 + float(centroids[label, 1])
        if abs(component_cy - expected_y) > y_tolerance:
            continue

        component = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        (refined_cx, refined_cy), refined_radius = cv2.minEnclosingCircle(contour)
        refined_radius = float(refined_radius)
        if refined_radius < max(7.0, config.min_radius * 0.45):
            continue

        center = (int(round(refined_cx + x1)), int(round(refined_cy + y1)))
        radius = int(round(refined_radius))
        radius = max(1, min(int(config.max_radius), radius))
        distance_score = max(
            0.0,
            1.0
            - float(np.hypot(center[0] - cx, center[1] - cy))
            / max(1.0, float(window_radius)),
        )
        y_score = max(0.0, 1.0 - abs(component_cy - expected_y) / y_tolerance)
        shape_score = aspect
        area_score = min(1.5, area / max(1.0, float(np.pi * config.min_radius * config.min_radius * 0.45)))
        enclosing_area = max(1.0, np.pi * refined_radius * refined_radius)
        fill_score = min(1.0, area / enclosing_area)

        # 球通常是候选区域内最大的近圆形高光块，且应靠近轨道中心线。
        score = (
            3.0 * y_score
            + 2.2 * shape_score
            + 1.1 * distance_score
            + 0.9 * fill_score
            + 0.8 * area_score
        )
        candidate = CircleInfo(center=center, radius=radius)
        if best is None or score > best[0]:
            best = (score, candidate)

    return best[1] if best is not None else None


def detect_ball(
    frame: np.ndarray,
    config: BallConfig = DEFAULT_CONFIG,
    track_info: Optional[TrackInfo] = None,
    manual_center: Optional[Point] = None,
    manual_radius: Optional[int] = None,
) -> Tuple[Optional[Point], Optional[int], np.ndarray]:
    """
    检测小球并返回 `(center, radius, visualization)`。

    检测顺序：
    1. 使用轨道 ROI 的右侧区域作为搜索范围；
    2. 灰度化 + 高斯滤波；
    3. HoughCircles 产生候选圆；
    4. 按半径、球心 y、圆周边缘支持度和右侧先验打分。

    如果传入 manual_center，则跳过自动检测；manual_radius 未提供时使用
    config.expected_radius。
    """

    height, width = frame.shape[:2]
    visualization = frame.copy()
    if track_info is not None:
        _draw_track_info(visualization, track_info)

    if manual_center is not None:
        radius = int(round(manual_radius if manual_radius is not None else config.expected_radius))
        radius = max(1, radius)
        cx = int(np.clip(manual_center[0], 0, width - 1))
        cy = int(np.clip(manual_center[1], 0, height - 1))
        cv2.circle(visualization, (cx, cy), radius, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(visualization, (cx, cy), 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            visualization,
            f"manual ball ({cx},{cy}) r={radius}",
            (max(5, cx - 100), max(22, cy - radius - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return (cx, cy), radius, visualization

    if config.ball_search_roi is not None:
        search_roi = _clip_roi(config.ball_search_roi, width, height)
    elif track_info is not None:
        search_x = max(track_info.x_start, int(round(width * config.ball_search_x_ratio)))
        search_y = max(0, track_info.top_y - 12)
        search_x2 = min(width, track_info.x_end + 1)
        search_y2 = min(height, track_info.bottom_y + 13)
        search_roi = (search_x, search_y, search_x2 - search_x, search_y2 - search_y)
    else:
        search_x = int(round(width * config.ball_search_x_ratio))
        search_y = int(round(height * 0.40))
        search_roi = (search_x, search_y, width - search_x, int(round(height * 0.22)))

    x0, y0, roi_width, roi_height = _clip_roi(search_roi, width, height)
    if roi_width < 16 or roi_height < 16:
        cv2.putText(
            visualization,
            "ball search ROI is invalid",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return None, None, visualization

    cv2.rectangle(visualization, (x0, y0), (x0 + roi_width - 1, y0 + roi_height - 1), (0, 165, 255), 2)

    crop = frame[y0 : y0 + roi_height, x0 : x0 + roi_width]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_kernel = _ensure_odd(config.gaussian_kernel)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), config.gaussian_sigma)
    edges = cv2.Canny(blurred, 40, 120)

    expected_y = track_info.center_y if track_info is not None else height * config.expected_y_ratio
    if track_info is not None:
        y_min = max(0, track_info.top_y - 12)
        y_max = min(height - 1, track_info.bottom_y + 12)
    else:
        y_min = max(0, int(round(expected_y - config.expected_y_tolerance)))
        y_max = min(height - 1, int(round(expected_y + config.expected_y_tolerance)))

    candidates: List[Tuple[float, Point, int, float]] = []
    # 第一轮使用配置值；检测困难时逐步降低 param2，但仍受尺寸和 y 范围约束。
    param2_values = [config.hough_param2, config.hough_param2 - 2.0, config.hough_param2 - 4.0]
    seen: List[Tuple[Point, int]] = []

    for param2 in param2_values:
        if param2 < 8:
            continue
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=config.hough_dp,
            minDist=config.hough_min_dist,
            param1=config.hough_param1,
            param2=param2,
            minRadius=config.min_radius,
            maxRadius=config.max_radius,
        )
        if circles is None:
            continue

        for raw_x, raw_y, raw_radius in np.round(circles[0]).astype(int):
            local_center = (int(raw_x), int(raw_y))
            center = (local_center[0] + x0, local_center[1] + y0)
            radius = int(raw_radius)
            cx, cy = center

            if not (config.min_radius <= radius <= config.max_radius):
                continue
            if not (x0 <= cx < x0 + roi_width and y0 <= cy < y0 + roi_height):
                continue
            if not (y_min <= cy <= y_max):
                continue
            if any(abs(cx - old_center[0]) < 5 and abs(cy - old_center[1]) < 5 for old_center, _ in seen):
                continue

            seen.append((center, radius))
            support = _ring_edge_support(edges, local_center, radius)
            y_scale = max(28.0, (y_max - y_min) * 0.30)
            y_score = max(0.0, 1.0 - abs(cy - expected_y) / y_scale)
            radius_scale = max(8.0, (config.max_radius - config.min_radius) / 2.0)
            radius_score = max(0.0, 1.0 - abs(radius - config.expected_radius) / radius_scale)
            right_score = (cx - x0) / max(1.0, roi_width - 1)
            score = 3.2 * y_score + 1.0 * radius_score + 0.8 * support + 0.15 * right_score
            candidates.append((score, center, radius, support))

    # 先绘制其它候选，再突出最终选择，便于调参时判断误检原因。
    for score, center, radius, support in sorted(candidates, reverse=True)[:20]:
        cv2.circle(visualization, center, radius, (0, 140, 255), 1, cv2.LINE_AA)
        cv2.putText(
            visualization,
            f"{score:.1f}",
            (center[0] + radius + 2, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 140, 255),
            1,
            cv2.LINE_AA,
        )

    if not candidates:
        cv2.putText(
            visualization,
            "Hough ball not found; use --ball x,y,r",
            (10, max(25, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return None, None, visualization

    _, hough_center, hough_radius, _ = max(candidates, key=lambda item: item[0])
    center = hough_center
    radius = hough_radius
    refined = _refine_reflective_circle(
        frame,
        seed_center=hough_center,
        seed_radius=hough_radius,
        expected_y=expected_y,
        config=config,
    )
    label_suffix = "hough"
    if refined is not None:
        center = refined.center
        radius = refined.radius
        label_suffix = "refined"
        # 紫色圆表示 Hough 的原始候选，绿色圆表示最终用于提取的球。
        cv2.circle(visualization, hough_center, hough_radius, (255, 0, 255), 2, cv2.LINE_AA)

    cv2.circle(visualization, center, radius, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(visualization, center, 3, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(
        visualization,
        f"ball ({center[0]},{center[1]}) r={radius} [{label_suffix}]",
        (max(5, center[0] - 105), max(22, center[1] - radius - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return center, radius, visualization


def extract_ball(
    frame: np.ndarray,
    circle_info: Union[CircleInfo, Tuple[Point, int]],
    feather_px: float = DEFAULT_CONFIG.feather_px,
    patch_margin: int = DEFAULT_CONFIG.patch_margin,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    提取小球局部图、圆形 mask 和透明背景图。

    返回：
        ball_patch: 固定大小的 BGR 局部图；
        ball_mask: 0/255 二值圆形 mask；
        ball_rgba: BGRA 图，alpha 边缘经过轻微羽化。
    """

    circle = _unpack_circle(circle_info)
    cx, cy = circle.center
    radius = max(1, int(circle.radius))
    margin = max(0, int(patch_margin))
    patch_radius = radius + margin
    patch_size = patch_radius * 2 + 1

    # 使用固定画布而不是直接裁剪，确保球心永远位于 patch 中心；
    # 即使小球靠近图像边缘，后续 compose 也不会发生中心偏移。
    ball_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    frame_height, frame_width = frame.shape[:2]
    src_x1 = max(0, cx - patch_radius)
    src_y1 = max(0, cy - patch_radius)
    src_x2 = min(frame_width, cx + patch_radius + 1)
    src_y2 = min(frame_height, cy + patch_radius + 1)
    dst_x1 = src_x1 - (cx - patch_radius)
    dst_y1 = src_y1 - (cy - patch_radius)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    if src_x2 > src_x1 and src_y2 > src_y1:
        ball_patch[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

    ball_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    cv2.circle(ball_mask, (patch_radius, patch_radius), radius, 255, -1, cv2.LINE_AA)

    if feather_px > 0:
        distance = cv2.distanceTransform((ball_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        alpha = np.clip(distance / float(feather_px), 0.0, 1.0) * 255.0
        alpha = alpha.astype(np.uint8)
    else:
        alpha = ball_mask.copy()

    ball_rgba = cv2.cvtColor(ball_patch, cv2.COLOR_BGR2BGRA)
    ball_rgba[:, :, 3] = alpha
    return ball_patch, ball_mask, ball_rgba


def remove_ball_from_background(
    frame: np.ndarray,
    circle_info: Union[CircleInfo, Tuple[Point, int]],
    track_info: Optional[TrackInfo] = None,
    inpaint_radius: float = DEFAULT_CONFIG.inpaint_radius,
    mask_dilate: int = DEFAULT_CONFIG.mask_dilate,
) -> np.ndarray:
    """
    用扩大后的圆形 mask 进行 Telea inpaint，生成去球背景。

    track_info 保留在接口中，方便后续加入轨道纹理拷贝策略；当前版本直接在
    原图上修复，能保留轨道两侧的边缘和颜色过渡。
    """

    del track_info  # 当前 inpaint 不需要额外读取轨道信息
    circle = _unpack_circle(circle_info)
    cx, cy = circle.center
    radius = max(1, int(circle.radius)) + max(0, int(mask_dilate))
    repair_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(repair_mask, (cx, cy), radius, 255, -1, cv2.LINE_AA)
    return cv2.inpaint(frame, repair_mask, float(max(1.0, inpaint_radius)), cv2.INPAINT_TELEA)


def _resize_ball_rgba(ball_rgba: np.ndarray, target_radius: Optional[int]) -> np.ndarray:
    """按 alpha 实际范围缩放小球，target_radius=None 时保持原尺寸。"""

    if target_radius is None or target_radius <= 0:
        return ball_rgba

    alpha = ball_rgba[:, :, 3]
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0 or len(ys) == 0:
        return ball_rgba

    source_radius = max(float(xs.max() - xs.min()), float(ys.max() - ys.min())) / 2.0
    if source_radius <= 1.0:
        return ball_rgba

    scale = float(target_radius) / source_radius
    if abs(scale - 1.0) < 0.01:
        return ball_rgba

    new_width = max(3, int(round(ball_rgba.shape[1] * scale)))
    new_height = max(3, int(round(ball_rgba.shape[0] * scale)))
    return cv2.resize(ball_rgba, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def _alpha_paste(base: np.ndarray, overlay: np.ndarray, center: Point) -> None:
    """把一个 BGRA patch 以 alpha 方式贴到 BGR base 上，支持边缘裁剪。"""

    overlay_height, overlay_width = overlay.shape[:2]
    center_x, center_y = center
    left = int(round(center_x - overlay_width / 2.0))
    top = int(round(center_y - overlay_height / 2.0))
    right = left + overlay_width
    bottom = top + overlay_height

    base_height, base_width = base.shape[:2]
    dst_x1 = max(0, left)
    dst_y1 = max(0, top)
    dst_x2 = min(base_width, right)
    dst_y2 = min(base_height, bottom)
    if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
        return

    src_x1 = dst_x1 - left
    src_y1 = dst_y1 - top
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    source = overlay[src_y1:src_y2, src_x1:src_x2]
    alpha = source[:, :, 3:4].astype(np.float32) / 255.0
    background = base[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
    foreground = source[:, :, :3].astype(np.float32)
    blended = foreground * alpha + background * (1.0 - alpha)
    base[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(blended, 0, 255).astype(np.uint8)


def compose_ball_on_track(
    background: np.ndarray,
    ball_rgba: np.ndarray,
    positions: Sequence[Position],
    track_info: TrackInfo,
    target_radius: Optional[int] = None,
) -> np.ndarray:
    """
    将一个小球合成到多个位置。

    positions 支持：
        [200, 350, 500]：x 坐标，y 自动取轨道中心线；
        [(200, 382), (350, 381)]：显式指定每个球心。
    """

    result = background.copy()
    overlay = _resize_ball_rgba(ball_rgba, target_radius)

    for item in positions:
        if isinstance(item, (tuple, list, np.ndarray)):
            x = int(round(float(item[0])))
            y = int(round(float(item[1])))
        else:
            x = int(round(float(item)))
            y = track_info.center_y_at(x)
        _alpha_paste(result, overlay, (x, y))

    return result


def save_debug_outputs(
    output_dir: Union[str, Path],
    detect_visualization: np.ndarray,
    ball_patch: np.ndarray,
    ball_mask: np.ndarray,
    ball_rgba: np.ndarray,
    background_without_ball: np.ndarray,
    composite_multi_ball: np.ndarray,
    sequence_frames: Optional[Sequence[np.ndarray]] = None,
) -> None:
    """保存全部中间结果和可选的逐帧结果。"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    outputs = {
        "debug_detect.png": detect_visualization,
        "extracted_ball_patch.png": ball_patch,
        "extracted_ball_mask.png": ball_mask,
        "extracted_ball.png": ball_rgba,
        "background_without_ball.png": background_without_ball,
        "composite_multi_ball.png": composite_multi_ball,
    }
    for filename, image in outputs.items():
        destination = output_path / filename
        if not cv2.imwrite(str(destination), image):
            raise IOError(f"无法写入输出文件: {destination}")

    if sequence_frames is not None:
        frames_path = output_path / "frames"
        frames_path.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(sequence_frames):
            destination = frames_path / f"frame_{index:03d}.png"
            if not cv2.imwrite(str(destination), image):
                raise IOError(f"无法写入序列帧: {destination}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 ROI + HoughCircles 提取不锈钢小球，并重组到轨道上的多个位置。"
    )
    parser.add_argument("--input", required=True, help="输入图像路径")
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        default="ball_image/artifacts/output",
        help="输出目录，默认 ball_image/artifacts/output",
    )
    parser.add_argument("--mode", choices=("multi", "sequence"), default="multi", help="multi 或 sequence")
    parser.add_argument(
        "--positions",
        default="200,350,500,650,800,950,1100",
        help="x 坐标列表，例如 200,350,500；也支持 x,y;x,y",
    )
    parser.add_argument("--position-y", type=int, default=None, help="统一指定所有目标位置的 y")
    parser.add_argument("--target-radius", type=int, default=None, help="合成时的目标半径，默认保持原半径")

    parser.add_argument("--ball", type=parse_circle, default=None, help="手动指定小球 x,y,r，例如 1238,382,17")
    parser.add_argument("--ball-center", type=parse_point, default=None, help="手动指定球心 x,y")
    parser.add_argument("--ball-radius", type=int, default=None, help="手动/自动检测的期望半径")
    parser.add_argument("--track-roi", type=parse_roi, default=None, help="手动指定轨道 x,y,w,h")
    parser.add_argument("--ball-search-roi", type=parse_roi, default=None, help="手动指定圆检测搜索 x,y,w,h")

    parser.add_argument("--min-radius", type=int, default=None, help="Hough 最小半径")
    parser.add_argument("--max-radius", type=int, default=None, help="Hough 最大半径")
    parser.add_argument("--hough-param1", type=float, default=None, help="Hough Canny 高阈值")
    parser.add_argument("--hough-param2", type=float, default=None, help="Hough 累加器阈值")
    parser.add_argument("--no-refine", action="store_true", help="关闭 Hough 后的反光球区域细化")
    parser.add_argument("--feather-px", type=float, default=None, help="小球 alpha 边缘羽化像素")
    parser.add_argument("--mask-dilate", type=int, default=None, help="背景修复 mask 外扩像素")
    parser.add_argument("--inpaint-radius", type=float, default=None, help="OpenCV inpaint 邻域半径")
    return parser


def _make_config(args: argparse.Namespace) -> BallConfig:
    values = {}
    if args.ball_radius is not None:
        values["expected_radius"] = float(args.ball_radius)
    if args.min_radius is not None:
        values["min_radius"] = int(args.min_radius)
    if args.max_radius is not None:
        values["max_radius"] = int(args.max_radius)
    if args.hough_param1 is not None:
        values["hough_param1"] = float(args.hough_param1)
    if args.hough_param2 is not None:
        values["hough_param2"] = float(args.hough_param2)
    values["refine_reflection"] = not args.no_refine
    if args.feather_px is not None:
        values["feather_px"] = float(args.feather_px)
    if args.mask_dilate is not None:
        values["mask_dilate"] = int(args.mask_dilate)
    if args.inpaint_radius is not None:
        values["inpaint_radius"] = float(args.inpaint_radius)
    values["ball_search_roi"] = args.ball_search_roi
    return replace(DEFAULT_CONFIG, **values)


def _load_image(path: Union[str, Path]) -> np.ndarray:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"输入图像不存在: {image_path}")
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"OpenCV 无法读取图像: {image_path}")
    return frame


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        frame = _load_image(args.input)
        config = _make_config(args)
        positions = parse_positions(args.positions)
        if args.position_y is not None:
            positions = [
                (int(item[0]) if isinstance(item, (tuple, list, np.ndarray)) else int(item), args.position_y)
                for item in positions
            ]

        track_info = detect_track_roi(frame, manual_roi=args.track_roi, config=config)

        manual_center: Optional[Point] = args.ball_center
        manual_radius: Optional[int] = args.ball_radius
        if args.ball is not None:
            manual_center = args.ball.center
            manual_radius = args.ball.radius

        center, radius, detection_visualization = detect_ball(
            frame,
            config=config,
            track_info=track_info,
            manual_center=manual_center,
            manual_radius=manual_radius,
        )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if center is None or radius is None:
            # 即使失败也保存调试图，方便根据 ROI 和候选圆调参。
            cv2.imwrite(str(output_dir / "debug_detect.png"), detection_visualization)
            print("自动检测失败，已保存 debug_detect.png。可使用 --ball x,y,r 手动指定。", file=sys.stderr)
            return 2

        circle_info = CircleInfo(center=center, radius=radius)
        ball_patch, ball_mask, ball_rgba = extract_ball(
            frame,
            circle_info,
            feather_px=config.feather_px,
            patch_margin=config.patch_margin,
        )
        background = remove_ball_from_background(
            frame,
            circle_info,
            track_info=track_info,
            inpaint_radius=config.inpaint_radius,
            mask_dilate=config.mask_dilate,
        )

        composite = compose_ball_on_track(
            background,
            ball_rgba,
            positions,
            track_info,
            target_radius=args.target_radius,
        )

        sequence_frames: Optional[List[np.ndarray]] = None
        if args.mode == "sequence":
            sequence_frames = []
            for position in positions:
                sequence_frames.append(
                    compose_ball_on_track(
                        background,
                        ball_rgba,
                        [position],
                        track_info,
                        target_radius=args.target_radius,
                    )
                )

        save_debug_outputs(
            output_dir=output_dir,
            detect_visualization=detection_visualization,
            ball_patch=ball_patch,
            ball_mask=ball_mask,
            ball_rgba=ball_rgba,
            background_without_ball=background,
            composite_multi_ball=composite,
            sequence_frames=sequence_frames,
        )

        print(f"输入图像: {args.input}")
        print(f"图像尺寸: {frame.shape[1]}x{frame.shape[0]}")
        print(f"轨道 ROI: {track_info.roi}, center_y={track_info.center_y:.1f}, method={track_info.method}")
        print(f"检测小球: center={center}, radius={radius}")
        print(f"重组位置: {positions}")
        print(f"输出目录: {output_dir.resolve()}")
        if sequence_frames is not None:
            print(f"序列帧: {output_dir / 'frames'} ({len(sequence_frames)} frames)")
        return 0
    except (FileNotFoundError, ValueError, IOError, cv2.error) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
