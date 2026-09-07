# YOLO references

本目录保存 Codex YOLO skills 按需读取的共享资料和子任务提示模板。

## 文件

| 文件 | 用途 |
|------|------|
| `yolo-env.md` | 项目目录、Python 环境、脚本入口、输出约定 |
| `yolo-validator.md` | 单模型验证子任务的输入、命令和输出契约 |
| `yolo-exporter.md` | 单格式导出子任务的输入、命令和输出契约 |

## 读取规则

先读取触发的 skill，再按其中的相对链接打开本目录下的必要 reference。不要为了一个训练或推理任务一次性读取全部 reference。

`yolo-validator.md` 和 `yolo-exporter.md` 是提示模板，不是独立的 Claude agent 配置；主 Codex 负责创建子任务、传入参数并汇总结果。

