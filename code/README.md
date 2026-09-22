# VAULT: anonymous release candidate

[中文说明](README.zh-CN.md)

This code repository contains **one** complete TEA + SEL + AEER /
Qwen3-VL-8B LoRA pipeline. The videos and annotations are distributed separately;
place them as described in [data/README.md](data/README.md). The dataset download
URL is pending. The code snapshot excludes original experiment history,
outputs, author accounts, original filenames, and source-path mappings. Existing
research directories are not used at runtime.

## Dataset

- 1,776 videos: 1,115 train / 356 validation / 305 test.
- 25 anomaly categories plus normal; annotations are multi-label.
- Four tasks: description, classification, temporal grounding, and VQA.
- 5,328 VQA questions; 6,690 training instructions (6 per training video).
- Videos: `data/videos/clip_000001.mp4`, etc. IDs are randomized and independent
  of category and split. The original-to-anonymous map is **not distributed**.
- Annotations: `data/annotations/{train,val,test}.jsonl`. `video_path` is relative
  to this release root. Semantic labels, descriptions, intervals, questions,
  answer options and answers are retained; annotator accounts and annotation
  timestamps are removed. Illumination scores are retained when available.
- `data/manifest.json` records sizes, SHA-256 checksums and media properties.

Videos are independent copies, not symlinks or hardlinks. MP4 user metadata,
private UUID metadata, container timestamps and handler names are cleared without
re-encoding. Video/audio sample data and temporal structures are preserved.
Embedded cover art stored as user metadata is removed. Visible scene content,
faces, signs, watermarks and speech are **not** anonymized by path/metadata cleanup.

## Installation

All commands below run from this `code/` directory. After downloading the
repository, enter it with `cd code` before installing dependencies or running scripts.

Use Python 3.11 and FFmpeg/ffprobe. Create a fresh environment; do not modify an
active experiment environment. Install a CUDA-compatible build of PyTorch and
then the training requirements:

```bash
python -m pip install -r requirements-train.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

The core training version pins are taken from the source environment. CUDA wheels,
FlashAttention and DeepSpeed still need to match your machine. A clean-machine GPU
installation and full training have not been validated by merely packaging this
snapshot. The included minimal Qwen visual utilities are loaded from the local
`qwen-vl-utils/src` directory. See `third_party/Qwen-LICENSE`.

Download **Qwen3-VL-8B-Instruct** into a local model directory. Model weights and
LoRA checkpoints are not included. Use an absolute local model path when invoking
launch scripts (example: `export BASE_MODEL="$PWD/models/Qwen3-VL-8B-Instruct"`).

## 1. Place and validate the separate dataset

After placing the separate dataset under `data/`, run from the repository root:

```bash
python scripts/audit_release.py --verify-hashes
python -m pip install -r requirements-eval.txt  # Also needed by the regression tests
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

## 2. Generate event streams

The experiments use **simulated v2e events**, not independently captured event
camera recordings. Event H5 caches are not bundled; regenerate them using an
installed [v2e](https://github.com/SensorsINI/v2e) checkout in its own environment.
This release wrapper follows the source pipeline's `--disable_slomo`, `--dvs240`,
`--dvs_params clean` conversion settings. It does not substitute optical flow or
fake events.

```bash
# V2E_ENTRY points to v2e.py; V2E_PYTHON selects its Python environment.
"$V2E_PYTHON" scripts/generate_events.py --split train --v2e "$V2E_ENTRY"
"$V2E_PYTHON" scripts/generate_events.py --split test --v2e "$V2E_ENTRY"
# Optional: also generate validation events using --split val.
```

Output: `data/events/<split>/<video_id>/<video_id>.h5` with dataset `events`,
columns `[timestamp_us, x, y, polarity]`, on a 240 × 180 event plane.
The wrapper's `--dry-run --limit 1` shows the conversion command without running
it. Exact simulated streams can depend on the v2e version and video decoder;
regeneration is not a guarantee of byte-identical historical event caches.

## 3. Prepare RGB/event training inputs

```bash
bash scripts/prepare.sh
```

This builds the four-task instructions from anonymous training annotations, runs
TEA/SEL with 16–24 requested observations and a 640-pixel maximum edge, and renders
AEER images with `sft/event_rgb/configs/adaptive_gray_pair.json`. Missing events
cause preparation to fail. Existing matching caches can be reused. Do not silently
truncate long examples; increase `MODEL_MAX_LENGTH` if preflight rejects a sample.

Generated RGB and event caches contain **local absolute paths** for runtime
integrity checks and are excluded from the release source by `.gitignore`.
Regenerate them after moving a prepared working tree. Never publish generated
caches or training outputs without an additional anonymization pass.

## 4. LoRA fine-tuning

```bash
export BASE_MODEL="$PWD/models/Qwen3-VL-8B-Instruct"
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash scripts/train.sh
```

Default: LoRA rank 32, alpha 64, dropout 0.05; learning rate 2e-5; 3 epochs;
effective batch 32; bf16; DeepSpeed ZeRO-2; model context 6144. The vision backbone
and projection module are frozen. The language-model attention and feed-forward
projections are tuned. No dedicated event encoder or auxiliary loss is added.
Override `OUTPUT_DIR`, `MODEL_MAX_LENGTH`, `NPROC_PER_NODE`, `BATCH_SIZE`, `GRAD_ACCUM`,
`MAX_STEPS`, and `REPORT_TO` using environment variables. `DRY_RUN=1` only prints
the launch command. It does not validate model tensors or perform training.

## 5. Merge a selected checkpoint

```bash
export ADAPTER="$PWD/outputs/lora/checkpoint-420"
bash scripts/merge.sh
```

Choose a checkpoint that actually exists; the step number above is an example,
not a provided result. Merging happens on CPU and requires sufficient RAM. The
merge script verifies the current input-policy signature and refuses to overwrite
an existing output. Default destination: `outputs/merged-model`.

## 6. Inference

Use a separate inference environment if necessary:

```bash
python -m pip install -r requirements-infer.txt
bash scripts/infer.sh --cuda-visible-devices 0 --tensor-parallel-size 1
```

This runs all four tasks and cyclic VQA option permutations. Results go to
`outputs/predictions/{d,c,t,vqa}.jsonl`. A CPU-only input smoke check is available:
`bash scripts/infer.sh --dry-run --max-samples 1`; it still reads videos/events,
but does not load model weights or produce meaningful predictions.

## 7. Evaluation

```bash
python -m pip install -r requirements-eval.txt
python -m nltk.downloader wordnet omw-1.4
bash scripts/evaluate.sh
```

Classification reports sample/micro/macro F1 over all 26 labels. Temporal grounding
uses interval-union IoU (normal/normal = 1; normal/anomaly mismatch = 0). VQA reports
standard accuracy and robustness over cyclic option permutations. Description
reports BLEU-4, METEOR, CIDEr and ROUGE-L. No external LLM API, API key or private
GPT evaluation prompt is required. Reports go to `outputs/metrics`.

## Scope and reproducibility

This release uses `timestamp_event_images_v3_reliable_segments` consistently for
new training data and inference. It does not promise exact reproduction of older
checkpoints trained using v2 caches or different budgets. See `VALIDATION.md` for
checks actually executed. This code snapshot does not imply that dataset publication or redistribution
licenses are finalized.

Publish the large video dataset separately from the code repository. `.gitignore`
intentionally excludes videos, event caches and model weights from ordinary Git.
Retain upstream notices; the project-specific code/data redistribution license
must be selected by the project owner before public release.
