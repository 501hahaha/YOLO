# -*- coding: utf-8 -*-
"""
生成不锈钢小球的 YOLO 检测训练数据集。

本脚本复用 ball_recompose.py 的检测、提取、背景修复和 alpha 合成逻辑，
在“去掉原始小球后的轨道背景”上随机放置 1 个或多个小球，生成：

    yolo_dataset/
    ├── data.yaml
    ├── classes.txt
    ├── images/train/*.jpg
    ├── images/val/*.jpg
    ├── images/test/*.jpg
    ├── labels/train/*.txt
    ├── labels/val/*.txt
    └── labels/test/*.txt

每一个 .txt 文件使用标准 YOLO 检测格式：

    class_id center_x center_y width height

其中坐标均已归一化到 0..1，类别 0 表示 ball。

示例：
    python scripts/dataset/generate_yolo_dataset.py \
        --input ball_image/source/canmv-frame-2026-09-05T10-59-25-573Z.png \
        --output_dir ball_image/datasets/yolo_dataset_10000 \
        --count 10000

自动检测失败时可以手动指定：
    python scripts/dataset/generate_yolo_dataset.py \
        --input test.png --output_dir yolo_dataset \
        --ball 1238,382,17 --track-roi 0,322,1280,108
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from ball_recompose import (
    DEFAULT_CONFIG,
    BallConfig,
    CircleInfo,
    Point,
    TrackInfo,
    compose_ball_on_track,
    detect_ball,
    detect_track_roi,
    extract_ball,
    parse_circle,
    parse_point,
    parse_roi,
    remove_ball_from_background,
)


@dataclass(frozen=True)
class DatasetConfig:
    """数据集随机合成参数。可直接修改这里，也可以使用命令行覆盖。"""

    count: int = 10000
    seed: int = 20260906

    # 数据集划分：默认 8000 train、1000 val、1000 test。
    train_ratio: float = 0.80
    val_ratio: float = 0.10

    # 每张合成图中的小球数量和相邻球的最小间距。
    min_balls: int = 1
    max_balls: int = 7
    min_spacing: int = 8
    y_jitter: int = 2
    edge_margin: int = 12

    # 以检测出的真实球半径为基准进行随机缩放，增加尺寸变化。
    radius_scale_min: float = 0.90
    radius_scale_max: float = 1.35

    # 默认保存 JPEG，兼顾训练质量和磁盘占用。
    image_format: str = "jpg"
    jpeg_quality: int = 92
    augment: bool = True


DEFAULT_DATASET_CONFIG = DatasetConfig()
Position = Tuple[int, int]


def _load_image(path: Union[str, Path]) -> np.ndarray:
    """读取输入图像并给出明确错误。"""

    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"输入图像不存在: {image_path}")
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"OpenCV 无法读取图像: {image_path}")
    return frame


def _validate_dataset_config(config: DatasetConfig) -> None:
    """检查数据集参数，避免生成过程中途失败。"""

    if config.count <= 0:
        raise ValueError("count 必须大于 0")
    if config.min_balls <= 0 or config.max_balls < config.min_balls:
        raise ValueError("必须满足 0 < min_balls <= max_balls")
    if config.train_ratio < 0 or config.val_ratio < 0:
        raise ValueError("train_ratio 和 val_ratio 不能为负数")
    if config.train_ratio + config.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio 必须小于 1，剩余部分用于 test")
    if config.radius_scale_min <= 0 or config.radius_scale_max < config.radius_scale_min:
        raise ValueError("半径缩放范围无效")
    if config.image_format.lower() not in {"jpg", "jpeg", "png"}:
        raise ValueError("image_format 只能是 jpg、jpeg 或 png")
    if not 1 <= config.jpeg_quality <= 100:
        raise ValueError("jpeg_quality 必须在 1..100 之间")


def _split_counts(count: int, train_ratio: float, val_ratio: float) -> Dict[str, int]:
    """计算 train/val/test 数量，并保证小数据集也能正常划分。"""

    if count == 1:
        return {"train": 1, "val": 0, "test": 0}
    if count == 2:
        return {"train": 1, "val": 0, "test": 1}

    train_count = int(round(count * train_ratio))
    val_count = int(round(count * val_ratio))
    train_count = max(1, min(count - 2, train_count))
    val_count = max(1, min(count - train_count - 1, val_count))
    return {
        "train": train_count,
        "val": val_count,
        "test": count - train_count - val_count,
    }


def _split_for_index(index: int, split_counts: Dict[str, int]) -> str:
    """按连续编号返回当前样本所属的 split。"""

    if index < split_counts["train"]:
        return "train"
    if index < split_counts["train"] + split_counts["val"]:
        return "val"
    return "test"


def _sample_positions(
    frame_shape: Tuple[int, ...],
    track_info: TrackInfo,
    rng: np.random.Generator,
    number: int,
    radius: int,
    y_jitter: int,
    min_spacing: int,
    edge_margin: int,
) -> List[Position]:
    """在轨道中心线附近随机采样不互相重叠的小球中心。"""

    height, width = frame_shape[:2]
    margin = max(0, int(edge_margin)) + radius
    x_min = max(0, track_info.x_start + margin)
    x_max = min(width - 1, track_info.x_end - margin)
    if x_max < x_min:
        x_min, x_max = radius, max(radius, width - radius - 1)

    y_jitter = max(0, int(y_jitter))
    min_distance = max(0, int(min_spacing)) + radius * 2
    positions: List[Position] = []

    # 随机尝试比目标数量多很多次，通常可以得到自然分布的位置。
    attempts = max(300, number * 300)
    for _ in range(attempts):
        x = int(rng.integers(x_min, x_max + 1))
        jitter = int(rng.integers(-y_jitter, y_jitter + 1)) if y_jitter else 0
        y = int(track_info.center_y_at(x) + jitter)
        if y - radius < 0 or y + radius >= height:
            continue
        if all(np.hypot(x - old_x, y - old_y) >= min_distance for old_x, old_y in positions):
            positions.append((x, y))
            if len(positions) >= number:
                break

    # 极端参数下随机采样可能不够，使用均匀位置作为可靠回退。
    if len(positions) < number:
        evenly_spaced = np.linspace(x_min, x_max, max(number, 2), dtype=np.int32)
        for x_value in evenly_spaced:
            x = int(x_value)
            y = int(track_info.center_y_at(x))
            if all(np.hypot(x - old_x, y - old_y) >= min_distance for old_x, old_y in positions):
                positions.append((x, y))
            if len(positions) >= number:
                break

    # 如果用户给了非常密集的数量/间距，最后允许降低间距，但仍不重复中心。
    if len(positions) < number:
        evenly_spaced = np.linspace(x_min, x_max, max(number, 2), dtype=np.int32)
        for x_value in evenly_spaced:
            x = int(x_value)
            y = int(track_info.center_y_at(x))
            if all(abs(x - old_x) >= 2 for old_x, _ in positions):
                positions.append((x, y))
            if len(positions) >= number:
                break

    return positions[:number]


def _augment_image(image: np.ndarray, rng: np.random.Generator, enabled: bool) -> np.ndarray:
    """做轻量相机风格扰动，标签不需要改变。"""

    if not enabled:
        return image

    result = image.astype(np.float32)

    # 亮度和对比度变化。
    contrast = float(rng.uniform(0.92, 1.08))
    brightness = float(rng.uniform(-9.0, 9.0))
    result = result * contrast + brightness
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 轻微色彩变化，模拟不同曝光/白平衡。
    if rng.random() < 0.65:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation_scale = float(rng.uniform(0.94, 1.06))
        value_scale = float(rng.uniform(0.96, 1.04))
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value_scale, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 小幅传感器噪声。
    if rng.random() < 0.45:
        sigma = float(rng.uniform(0.3, 1.8))
        noise = rng.normal(0.0, sigma, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 少量模糊样本，避免模型只适应绝对清晰的合成图。
    if rng.random() < 0.10:
        result = cv2.GaussianBlur(result, (3, 3), 0.0)

    return result


def _yolo_line(center: Position, radius: int, width: int, height: int) -> str:
    """将圆形目标转换为 YOLO 归一化矩形框。"""

    cx, cy = center
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(width, cx + radius)
    y2 = min(height, cy + radius)
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    box_center_x = (x1 + x2) / 2.0
    box_center_y = (y1 + y2) / 2.0
    return (
        f"0 {box_center_x / width:.6f} {box_center_y / height:.6f} "
        f"{box_width / width:.6f} {box_height / height:.6f}"
    )


def _write_image(path: Path, image: np.ndarray, jpeg_quality: int) -> None:
    """保存图片并检查 OpenCV 返回值。"""

    params: List[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    if not cv2.imwrite(str(path), image, params):
        raise IOError(f"无法写入图像: {path}")


def _prepare_dataset_tree(output_dir: Path) -> None:
    """创建 YOLO 标准目录。"""

    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_dataset_description(
    output_dir: Path,
    source_image: Optional[Union[str, Path]],
    circle_info: CircleInfo,
    track_info: TrackInfo,
    config: DatasetConfig,
    split_counts: Dict[str, int],
) -> None:
    """写入 Ultralytics/YOLO 可直接使用的配置和说明文件。"""

    (output_dir / "data.yaml").write_text(
        "# YOLO dataset generated by generate_yolo_dataset.py\n"
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: ball\n",
        encoding="utf-8",
    )
    (output_dir / "classes.txt").write_text("ball\n", encoding="utf-8")

    source_text = str(source_image) if source_image is not None else "<in-memory frame>"
    readme = f"""# Stainless-steel ball YOLO dataset

- class 0: `ball`
- images: `{config.image_format}`
- requested/generated images: `{config.count}`
- split: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}
- source image: `{source_text}`
- source ball: center={circle_info.center}, radius={circle_info.radius}
- track ROI: `{track_info.roi}`, center_y={track_info.center_y:.2f}

Ultralytics training example:

```bash
yolo detect train data=data.yaml model=yolov8n.pt imgsz=640 epochs=100
```

Each label line is:

```text
class_id center_x center_y width height
```

All coordinates are normalized to 0..1.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def generate_yolo_dataset(
    frame: np.ndarray,
    output_dir: Union[str, Path],
    circle_info: CircleInfo,
    track_info: TrackInfo,
    *,
    source_image: Optional[Union[str, Path]] = None,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    ball_config: BallConfig = DEFAULT_CONFIG,
) -> Dict[str, int]:
    """生成合成图片、YOLO 标签和数据集配置文件。

    该函数也可以从其它 Python 程序直接调用。每张图先从无球背景开始，
    再随机放置多个球，最后做轻量全图增强；标签根据实际合成位置生成。
    """

    _validate_dataset_config(dataset_config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _prepare_dataset_tree(output_path)

    split_counts = _split_counts(
        dataset_config.count,
        dataset_config.train_ratio,
        dataset_config.val_ratio,
    )
    _write_dataset_description(
        output_path,
        source_image,
        circle_info,
        track_info,
        dataset_config,
        split_counts,
    )

    # 保存合成所依据的中间资产，便于以后复现和调参。
    ball_patch, ball_mask, ball_rgba = extract_ball(
        frame,
        circle_info,
        feather_px=ball_config.feather_px,
        patch_margin=ball_config.patch_margin,
    )
    background = remove_ball_from_background(
        frame,
        circle_info,
        track_info=track_info,
        inpaint_radius=ball_config.inpaint_radius,
        mask_dilate=ball_config.mask_dilate,
    )
    _write_image(output_path / "source_ball_patch.png", ball_patch, dataset_config.jpeg_quality)
    _write_image(output_path / "source_ball_mask.png", ball_mask, dataset_config.jpeg_quality)
    _write_image(output_path / "source_ball.png", ball_rgba, dataset_config.jpeg_quality)
    _write_image(output_path / "source_background_without_ball.png", background, dataset_config.jpeg_quality)

    rng = np.random.default_rng(dataset_config.seed)
    image_suffix = ".jpg" if dataset_config.image_format.lower() in {"jpg", "jpeg"} else ".png"
    generated_counts = {"train": 0, "val": 0, "test": 0}
    start_time = time.perf_counter()
    height, width = frame.shape[:2]

    for index in range(dataset_config.count):
        split = _split_for_index(index, split_counts)
        target_radius = max(
            3,
            int(
                round(
                    circle_info.radius
                    * rng.uniform(
                        dataset_config.radius_scale_min,
                        dataset_config.radius_scale_max,
                    )
                )
            ),
        )
        ball_number = int(rng.integers(dataset_config.min_balls, dataset_config.max_balls + 1))
        positions = _sample_positions(
            frame.shape,
            track_info,
            rng,
            number=ball_number,
            radius=target_radius,
            y_jitter=dataset_config.y_jitter,
            min_spacing=dataset_config.min_spacing,
            edge_margin=dataset_config.edge_margin,
        )
        if not positions:
            raise RuntimeError("无法在当前轨道 ROI 内生成目标位置，请减小球数量或半径")

        composite = compose_ball_on_track(
            background,
            ball_rgba,
            positions,
            track_info,
            target_radius=target_radius,
        )
        composite = _augment_image(composite, rng, dataset_config.augment)

        stem = f"ball_{index:06d}"
        image_path = output_path / "images" / split / f"{stem}{image_suffix}"
        label_path = output_path / "labels" / split / f"{stem}.txt"
        _write_image(image_path, composite, dataset_config.jpeg_quality)
        label_path.write_text(
            "\n".join(_yolo_line(position, target_radius, width, height) for position in positions) + "\n",
            encoding="utf-8",
        )
        generated_counts[split] += 1

        done = index + 1
        if done % 250 == 0 or done == dataset_config.count:
            elapsed = max(0.001, time.perf_counter() - start_time)
            rate = done / elapsed
            print(
                f"generated {done}/{dataset_config.count} "
                f"({rate:.1f} images/s)",
                flush=True,
            )

    metadata = {
        "class_names": ["ball"],
        "count_requested": dataset_config.count,
        "count_generated": sum(generated_counts.values()),
        "split_counts": generated_counts,
        "seed": dataset_config.seed,
        "image_format": dataset_config.image_format,
        "source_image": str(source_image) if source_image is not None else None,
        "source_ball": {
            "center": list(circle_info.center),
            "radius": circle_info.radius,
        },
        "track": {
            "roi": list(track_info.roi),
            "center_y": track_info.center_y,
            "top_y": track_info.top_y,
            "bottom_y": track_info.bottom_y,
            "method": track_info.method,
        },
        "generation": {
            "min_balls": dataset_config.min_balls,
            "max_balls": dataset_config.max_balls,
            "min_spacing": dataset_config.min_spacing,
            "y_jitter": dataset_config.y_jitter,
            "radius_scale_min": dataset_config.radius_scale_min,
            "radius_scale_max": dataset_config.radius_scale_max,
            "augmentation": dataset_config.augment,
        },
    }
    (output_path / "dataset_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return generated_counts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="随机合成不锈钢小球图像，并生成标准 YOLO 检测数据集。"
    )
    parser.add_argument("--input", required=True, help="输入原始轨道图像")
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        default="ball_image/datasets/yolo_dataset_10000",
        help="YOLO 数据集输出目录",
    )
    parser.add_argument("--count", type=int, default=10000, help="生成图片数量，默认 10000")
    parser.add_argument("--seed", type=int, default=20260906, help="随机种子")
    parser.add_argument("--min-balls", type=int, default=1, help="每张图最少小球数")
    parser.add_argument("--max-balls", type=int, default=7, help="每张图最多小球数")
    parser.add_argument("--min-spacing", type=int, default=8, help="球心之间额外最小间距")
    parser.add_argument("--y-jitter", type=int, default=2, help="球心 y 方向随机扰动")
    parser.add_argument("--edge-margin", type=int, default=12, help="轨道两端安全边距")
    parser.add_argument("--radius-scale-min", type=float, default=0.90, help="最小半径缩放")
    parser.add_argument("--radius-scale-max", type=float, default=1.35, help="最大半径缩放")
    parser.add_argument("--image-format", choices=("jpg", "png"), default="jpg", help="输出图片格式")
    parser.add_argument("--jpeg-quality", type=int, default=92, help="JPEG 质量 1..100")
    parser.add_argument("--no-augment", action="store_true", help="关闭亮度、噪声和轻微模糊增强")

    parser.add_argument("--ball", type=parse_circle, default=None, help="手动指定小球 x,y,r，例如 1238,382,17")
    parser.add_argument("--ball-center", type=parse_point, default=None, help="手动指定球心 x,y")
    parser.add_argument("--ball-radius", type=int, default=None, help="手动/自动期望半径")
    parser.add_argument("--track-roi", type=parse_roi, default=None, help="手动指定轨道 x,y,w,h")
    parser.add_argument("--ball-search-roi", type=parse_roi, default=None, help="手动指定球检测搜索 x,y,w,h")
    parser.add_argument("--min-radius", type=int, default=None, help="Hough 最小半径")
    parser.add_argument("--max-radius", type=int, default=None, help="Hough 最大半径")
    parser.add_argument("--hough-param1", type=float, default=None, help="Hough Canny 高阈值")
    parser.add_argument("--hough-param2", type=float, default=None, help="Hough 累加器阈值")
    return parser


def _make_ball_config(args: argparse.Namespace) -> BallConfig:
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
    values["ball_search_roi"] = args.ball_search_roi
    return replace(DEFAULT_CONFIG, **values)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        frame = _load_image(args.input)
        ball_config = _make_ball_config(args)
        dataset_config = DatasetConfig(
            count=args.count,
            seed=args.seed,
            min_balls=args.min_balls,
            max_balls=args.max_balls,
            min_spacing=args.min_spacing,
            y_jitter=args.y_jitter,
            edge_margin=args.edge_margin,
            radius_scale_min=args.radius_scale_min,
            radius_scale_max=args.radius_scale_max,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
            augment=not args.no_augment,
        )

        track_info = detect_track_roi(frame, manual_roi=args.track_roi, config=ball_config)
        manual_center: Optional[Point] = args.ball_center
        manual_radius: Optional[int] = args.ball_radius
        if args.ball is not None:
            manual_center = args.ball.center
            manual_radius = args.ball.radius

        center, radius, detection_visualization = detect_ball(
            frame,
            config=ball_config,
            track_info=track_info,
            manual_center=manual_center,
            manual_radius=manual_radius,
        )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_image(output_dir / "debug_detect.png", detection_visualization, dataset_config.jpeg_quality)
        if center is None or radius is None:
            print("自动检测失败，已保存 debug_detect.png；请使用 --ball x,y,r 重试。", file=sys.stderr)
            return 2

        circle_info = CircleInfo(center=center, radius=radius)
        counts = generate_yolo_dataset(
            frame,
            output_dir,
            circle_info,
            track_info,
            source_image=args.input,
            dataset_config=dataset_config,
            ball_config=ball_config,
        )
        print(f"input: {args.input}")
        print(f"image: {frame.shape[1]}x{frame.shape[0]}")
        print(f"track: roi={track_info.roi}, center_y={track_info.center_y:.1f}, method={track_info.method}")
        print(f"ball: center={circle_info.center}, radius={circle_info.radius}")
        print(f"dataset: {output_dir.resolve()}")
        print(f"split: {counts}")
        print("YOLO config: data.yaml")
        return 0
    except (FileNotFoundError, ValueError, IOError, RuntimeError, cv2.error) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
