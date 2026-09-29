# EventVAULT

The runnable project is in [`code/`](code/). Run `cd code` before the commands below.

TEA + SEL + AEER for Qwen3-VL-8B LoRA training and four-task inference.
Training and inference share `configs/event_config.json`.

## 1. Install dependencies and place the model

Use the Python 3.11 `qwen2` environment for training, inference, and evaluation,
including description metrics.

```bash
conda activate qwen2
python -m pip install --no-build-isolation -r requirements.txt
python -m nltk.downloader wordnet omw-1.4
```

Place the model weights, tokenizer, and processor in
`models/Qwen3-VL-8B-Instruct/`. Run the following commands from the repository root.

## 2. Download videos and annotations separately

This repository contains code only. Download the RGB videos and annotation split
files from the separate EventVAULT dataset repository before preparing training
inputs or running inference. The dataset repository link will be added after the
dataset is released.

Place the downloaded files under the code repository's local `data/` directory:

```bash
mkdir -p data
cp -a /path/to/downloaded-dataset/videos /path/to/downloaded-dataset/annotations data/
```

The resulting local layout is:

```text
data/
  videos/clip_000001.mp4 ...
  annotations/train.jsonl
  annotations/val.jsonl
  annotations/test.jsonl
  events/train/clip_000001/clip_000001.h5 ...
  events/test/clip_000002/clip_000002.h5 ...
```

Annotation `video_path` values are relative to the code repository root, such as
`data/videos/clip_000001.mp4`. Each JSONL record provides
`video_id`, `video_path`, `category`, `description`, `temporal_segments`, and
`vqa_pairs`. Time intervals are in seconds; normal videos use `[[-1, -1]]`.
Each VQA pair contains `question`, four `options` keyed A/B/C/D, and `correct_key`.

Training uses `data/annotations/train.jsonl`; inference uses
`data/annotations/test.jsonl`. The local `data/` directory is ignored by Git.

After downloading the RGB videos and annotations, place matching event H5 files
under `data/events/train/` and `data/events/test/`. Event H5 files contain an
`events` array with columns
`[timestamp_us, x, y, polarity]`, aligned to the RGB source clock and a 240 x 180
sensor. Supply matching H5 files or generate them with an installed v2e simulator
and FFmpeg:

```bash
python scripts/generate_events.py --split train --v2e /path/to/v2e.py
python scripts/generate_events.py --split test --v2e /path/to/v2e.py
```

## 3. Prepare training inputs

```bash
WORKERS=2 bash scripts/prepare.sh
```

This builds four-task instructions, selects TEA/SEL RGB observations, renders
AEER event images, and assembles the final inputs. The training file is
`sft/data/all_tasks_mixed_eventvault.json`. Keep the generated RGB arrays and
event PNGs under `sft/cache/`, which this JSON references.

Edit `configs/event_config.json` to change AEER settings. Defaults use up to three
score-ranked anchor groups, three images per group, and adaptive windows of
3/6/12/24/48 source frames. RGB inputs use 16–24 blocks, duplicated into temporal
pairs; maximum RGB edge is 352 during preparation and 640 during inference.

## 4. Train and merge LoRA

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train.sh
ADAPTER="$PWD/outputs/lora/checkpoint-420" bash scripts/merge.sh
```

Edit training parameters in `sft/script/sft_8b_eventvault.sh`. Defaults use two GPU
processes, batch 1 per GPU, accumulation 16 (effective batch 32), two epochs,
AdamW, learning rate `2e-5`, warm-up ratio `0.1`, cosine decay, LoRA rank 32,
alpha 64, and dropout 0.05. Choose the desired checkpoint in the merge command.
Merged weights are saved to `outputs/merged-model/`.

## 5. Infer

```bash
bash scripts/infer.sh --cuda-visible-devices 0
```

Outputs are `outputs/predictions/{d,c,t,vqa}.jsonl`, for description,
classification, temporal localization, and VQA. Each VQA question is evaluated
with four circular option shifts. Inference resumes existing outputs; use a new
`PREDICTIONS` directory when changing the model or sampling configuration.

## 6. Evaluate

```bash
bash scripts/evaluate.sh
bash scripts/evaluate.sh --caption-metrics

# Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL), then score descriptions:
python eval_script/eval_des_gpt.py --input outputs/predictions/d.jsonl
bash scripts/evaluate.sh --judge-scores outputs/metrics/description_gpt.csv
```

`outputs/metrics/summary.json` reports mIoU, sample F1, standard VQA accuracy,
and robust VQA accuracy. Robust accuracy requires all four shifts to be correct.
IoU/F1/VQA use [0,1]; S_GPT uses [0,100]. The judge prompts are in
`configs/S_GPT_prompts.json`. S_GPT is computed from the five dimension means:

```text
(2.5*semantic + 2*detail + 2*causality + 3.5*saliency) * (1 + hallucination/20)
```

Qwen-derived source retains its upstream license in `LICENSE-Qwen` and
attribution in `NOTICE.md`.
