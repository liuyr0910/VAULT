# Validation of this local release candidate

Checked on 2026-09-21 against the complete local anonymous code-and-data release.
This upload contains only its code; dataset-dependent checks require the separate
dataset. These are packaging/integration checks, not model results.

## Completed

- All 1,776 anonymous videos are independent copies. Every source/copy pair was
  compared with ffprobe: video/audio codec, dimensions, frame count, frame rate,
  time base, start time, stream duration and container duration are preserved.
  Embedded cover art is intentionally excluded from that stream comparison.
- All 1,776 output SHA-256 values were independently recomputed against
  `data/manifest.json`; split inventories and file sizes matched.
- All original semantic annotation fields (categories, descriptions, intervals,
  environment information and VQA) were compared with the corresponding anonymous
  records and matched. No overlapping video IDs across train/validation/test.
- The full training instruction builder produced 6,690 records from 1,115 videos.
- Text scan found no source account names, annotator usernames, personal absolute
  paths or credential patterns. The private search terms and source map are kept
  outside this release. No Git history, symlinks or runtime caches are shipped.
- Python syntax and Bash syntax passed for all shipped code.
- Five CPU regression tests passed: complete instruction/path contract; all 26
  classification labels; temporal normal/overlap cases; incomplete circular VQA;
  rejection of modified event-cache configuration.
- A real anonymous training video with its matching existing v2e event stream ran
  through TEA/SEL preparation and AEER rendering: 6 instructions, 16 RGB
  observations, 9 event images. Temporary caches were outside this release.
- Qwen3-VL's actual processor checked 3 of those records on CPU; all 6 were
  checked for sequence length (maximum 3,468 tokens, below 6,144). RGB tensors,
  RGB RoPE positions and supervised answer labels matched the RGB baseline;
  disabling the supplement reproduced baseline input IDs exactly.
- Training entry-point imports/argument parsing and shell dry-run passed. No
  model weights were loaded for training and no training step was executed.
- All-task inference dry-run consumed the same real RGB/event sample and wrote
  1 description, 1 classification, 1 grounding and 12 circular VQA records.
- All four evaluation CLIs completed on exact-answer fixtures with the inference
  output schema. Those fixture scores are **not** experimental model performance.

## Compatibility fixes in this snapshot

The weight merger now checks the same v3 input policy used by preparation and
inference. Classification includes all 26 dataset labels (the previous evaluator
omitted drowning and animal aggression); its macro metrics therefore need not
match older reports. Temporal parsing recognizes the normal sentinel `-1 - -1`.
The VQA per-question CSV no longer marks an incomplete circular group as robust.

## Not yet validated

A fresh-machine dependency installation, regeneration of every v2e stream, full
GPU fine-tuning, merging an actual trained adapter, GPU/vLLM generation and full
benchmark inference have not been run for this release. Existing experiments
were not changed. Re-run preflight for the entire prepared training set before
training; a one-video check does not establish that every example fits context.

Path/metadata anonymization does not anonymize visible people, signs, watermarks,
or speech. Redistributable model weights and the large event caches are not
included. The project owner still needs to select the code/data release licenses.
