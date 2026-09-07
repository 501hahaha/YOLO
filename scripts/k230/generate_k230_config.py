"""从最终 YOLOv5 checkpoint 生成 K230 AnchorBaseDet 部署配置。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


# YOLOv5 checkpoint 反序列化时需要找到项目内的 models 包，脚本可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = PROJECT_ROOT / "yolov5-7.0"
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))


def load_model(checkpoint_path: Path):
    """加载包含完整模型对象的 YOLOv5 checkpoint。"""
    # PyTorch 2.6+ 默认 weights_only=True；YOLOv5 checkpoint 需要完整模型对象。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        raise ValueError(f"checkpoint 中没有 model/ema: {checkpoint_path}")
    if hasattr(model, "module"):
        model = model.module
    return model


def get_pixel_anchors(model):
    """读取 Detect 层的 grid anchors，并还原为输入像素单位。"""
    detect = model.model[-1]
    if not hasattr(detect, "anchors") or not hasattr(detect, "stride"):
        raise ValueError("checkpoint 不是带 Detect 层的 YOLOv5 检测模型")
    anchors = detect.anchors.detach().float().cpu()
    stride = detect.stride.detach().float().cpu().view(-1, 1, 1)
    pixel_anchors = anchors * stride
    return [
        [int(round(value)) for value in layer.reshape(-1).tolist()]
        for layer in pixel_anchors
    ]


def get_categories(model, fallback: str):
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]
    if names is None:
        names = [fallback]
    names = [str(name) for name in names]
    if len(names) != 1:
        raise ValueError(f"当前配置只支持单类别，但 checkpoint 有 {len(names)} 个类别: {names}")
    return names


def main():
    parser = argparse.ArgumentParser(description="Generate K230 deploy_config.json from YOLOv5 checkpoint")
    parser.add_argument("--weights", required=True, type=Path, help="YOLOv5 best.pt")
    parser.add_argument("--output", required=True, type=Path, help="输出 deploy_config.json")
    parser.add_argument("--model-name", default="ball_yolov5n_320.kmodel")
    parser.add_argument("--class-name", default="ball")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--nncase-version", default="2.9.0")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    model = load_model(args.weights)
    categories = get_categories(model, args.class_name)
    config = {
        "chip_type": "k230",
        "inference_width": 320,
        "inference_height": 320,
        "confidence_threshold": args.confidence,
        "nms_threshold": args.nms,
        "nncase_version": args.nncase_version,
        "model_type": "AnchorBaseDet",
        "img_size": [320, 320],
        "anchors": get_pixel_anchors(model),
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "categories": categories,
        "nms_option": False,
        "kmodel_path": args.model_name,
        "num_classes": len(categories),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
