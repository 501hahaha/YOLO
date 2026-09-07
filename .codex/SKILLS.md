# Project Codex Skills

本目录是本项目的 Codex skill 入口。`.claude/skills/` 和 `.claude/references/` 保留为源副本，`.codex/` 是去掉 Claude 专用字段并调整为 Codex 路径后的副本。

## 当前 skills

- `yolo-train`：YOLOv5 v7.0 本地训练、断点续训和参数调整
- `yolo-validate`：验证集评估和模型指标对比
- `yolo-export`：YOLOv5 模型导出为 ONNX 等格式
- `yolo-kmodel`：使用 `K230_Yolov5n` 将 ONNX 转换为 K230 kmodel，并用 CPU reference simulator 做数值验证
- `yolo-pipeline`：训练、验证、导出和 K230 转换的完整流程
- `yolo-detect`：图片、视频和摄像头推理
- `yolo-label`：X-AnyLabeling 数据标注流程
- `yolo-train-server`：SSH/rsync 远程训练

## 共享参考

- `references/yolo-env.md`：本项目路径、环境和输出约定
- `references/yolo-exporter.md`：模型导出任务约定
- `references/yolo-validator.md`：模型验证任务约定
- `references/README.md`：reference 读取规则

## 维护规则

1. 修改 `.claude` 源 skill 后，同步更新 `.codex` 副本；Codex 副本不复制 `allowed-tools` 等 Claude 专用字段。
2. 共享路径使用项目相对路径，不能写回失效的 `.claude` 路径。
3. 修改 `SKILL.md` 后检查 UTF-8、YAML frontmatter、命名和 `git diff --check`。
4. 当前 K230 转换的具体参数以 `skills/yolo-kmodel/SKILL.md` 和 `docs/YOLO_K230_WORKFLOW.md` 为准。
