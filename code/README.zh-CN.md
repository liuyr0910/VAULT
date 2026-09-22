# VAULT 匿名开源副本

这是独立的代码发布副本，整理的是 **TEA＋SEL＋AEER / timestamp-event-images
LoRA** 版本。视频、标注和生成的训练指令不包含在代码仓库中；数据集下载地址待补充，
放置方式见 [data/README.md](data/README.md)。下表的数据路径表示下载后的预期位置。完整命令见 [README.md](README.md)，已执行的验证见
[VALIDATION.md](VALIDATION.md)。

## 文件入口

| 内容 | 位置 |
| --- | --- |
| 全部 1776 个匿名视频（约 34.7 GiB） | `data/videos/` |
| 训练／验证／测试标注 | `data/annotations/{train,val,test}.jsonl` |
| 视频清单与 SHA-256 | `data/manifest.json` |
| 运行脚本后生成的 6690 条训练指令 | `sft/data/all_tasks_mixed.json` |
| 从视频生成 v2e 事件 | `scripts/generate_events.py` |
| TEA/SEL 采样与 AEER 图像准备 | `scripts/prepare.sh` |
| LoRA 微调 | `scripts/train.sh` |
| 选定 checkpoint 的权重合并 | `scripts/merge.sh` |
| 四任务推理 | `scripts/infer.sh` |
| 四任务评测 | `scripts/evaluate.sh` |
| 完整性与隐私扫描 | `scripts/audit_release.py` |

划分保留为训练 1115、验证 356、测试 305；共 5328 个 VQA 问题。
原始类别、描述、时间段、环境说明和问答内容保持不变。

## 匿名化范围

视频改成随机分配的 `clip_000001.mp4` 一类编号；标注使用相对路径，
删除标注者账号与操作时间；清除视频容器中的用户元数据和创建／修改时间。
视频没有重新编码，时间轴经过逐文件核对。没有复制 Git 历史、私有账号、
原始文件名映射、实验日志或模型权重。

原实验目录独立保留；这个代码目录用于匿名代码发布。
画面中的人物、标识、水印和音频内容未做匿名化处理。

## 使用与发布

在副本根目录按英文说明创建独立环境，并依次执行事件生成、数据准备、
训练、合并、推理和评测。默认基座为本地下载的 Qwen3-VL-8B-Instruct。
约 102 GiB 的原实验事件缓存未复制，提供 v2e 生成入口和对应配置。

测试已覆盖全量视频校验、标注核对、隐私扫描和一个真实样本的数据流程；
未运行完整 GPU 训练、实际权重合并或完整基准评测。

训练与推理运行后生成的缓存、模型配置和结果可能包含运行机器的绝对路径。
它们已列入 `.gitignore`，再次发布时应重新审查。视频也默认不加入普通 Git，
可作为独立数据集发布。副本外的原始路径映射只供本地追溯，不应公开。
