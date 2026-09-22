# Dataset placement

The dataset is distributed separately. Its download URL has not yet been added.
Place the anonymous dataset under this directory before preparation or training:

```text
data/
  annotations/train.jsonl
  annotations/val.jsonl
  annotations/test.jsonl
  videos/clip_000001.mp4
  videos/...
  manifest.json
```

There are 1,776 videos (1,115 train / 356 validation / 305 test).
Annotation video paths are relative to the project root.
Run `python scripts/build_sft.py` to generate the training instruction JSON;
it is not bundled in this code repository.
Run `python scripts/audit_release.py --verify-hashes` after placing the dataset.
Generate event streams using `scripts/generate_events.py` as described in the
root README. Video data, annotations, generated instructions and event caches
are excluded from code uploads.
