# VAULT

Research code for **TEA + SEL + AEER**, using the timestamp-event-images
Qwen3-VL-8B LoRA pipeline.

## Code

The implementation is in [`code/`](code/README.md):

- [Installation and full workflow](code/README.md)
- [中文说明](code/README.zh-CN.md)
- [Data preparation](code/scripts/prepare.sh)
- [LoRA training](code/scripts/train.sh)
- [Weight merging](code/scripts/merge.sh)
- [Inference](code/scripts/infer.sh)
- [Evaluation](code/scripts/evaluate.sh)
- [Validation scope](code/VALIDATION.md)

Run all workflow commands from `code/`:

```bash
cd code
```

## Dataset

Videos and annotations are distributed separately; the download link is pending.
See [dataset placement](code/data/README.md) for the expected directory structure.
Dataset videos, annotations, event caches and model checkpoints are not included
in this code repository.

## Repository layout

`index.html` and `static/` serve the project website. The `code/` directory is the
standalone training and evaluation package.
