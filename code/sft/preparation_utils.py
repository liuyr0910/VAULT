"""Shared instruction and cache helpers."""

import json
import re
from pathlib import Path

VIDEO_TAG = re.compile(r"<video>(.*?)</video>", flags=re.DOTALL)


def load_records(path):
    return json.loads(Path(path).read_text())


def video_path_from_record(record):
    if record.get("video"):
        value = record["video"]
        return str(Path(value if isinstance(value, str) else value[0]).resolve())
    text = next(
        t["value"] for t in record["conversations"] if t["from"] in ("user", "human")
    )
    return str(Path(VIDEO_TAG.search(text).group(1)).resolve())


def unique_video_paths(records):
    return list(dict.fromkeys(video_path_from_record(record) for record in records))


def save_json(value, path, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=indent) + "\n")
