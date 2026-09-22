from __future__ import annotations
import json, os, re, hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Iterable
VIDEO_TAG = re.compile(r"<video>(.*?)</video>", flags=re.DOTALL)

def load_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"Expected a JSON list in {path}")
    return records

def video_path_from_record(record: Dict) -> str:
    paths: List[str] = []
    explicit = record.get("video") or []
    if isinstance(explicit, str):
        explicit = [explicit]
    paths.extend(str(path).strip() for path in explicit if str(path).strip())

    for turn in record.get("conversations", []):
        if turn.get("from") not in ("user", "human"):
            continue
        paths.extend(match.strip() for match in VIDEO_TAG.findall(str(turn.get("value", ""))))

    unique = list(dict.fromkeys(path for path in paths if path))
    if len(unique) != 1:
        raise ValueError(
            f"Record {record.get('id', '<unknown>')} must reference exactly one video; got {unique}"
        )
    return str(Path(unique[0]).expanduser().resolve())

def unique_video_paths(records: Sequence[Dict]) -> List[str]:
    return list(dict.fromkeys(video_path_from_record(record) for record in records))

def file_fingerprint(path: Optional[str]) -> Optional[Dict]:
    if not path:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return None
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }

def stable_video_key(video_path: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(video_path).stem).strip("._") or "video"
    digest = hashlib.sha256(video_path.encode("utf-8")).hexdigest()[:12]
    return f"{stem}-{digest}"

def atomic_json_dump(value, path: Path, *, indent: Optional[int] = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")
    os.replace(temporary, path)

def task_name(record: Dict) -> str:
    identifier = str(record.get("id", ""))
    if identifier.endswith("_desc"):
        return "description"
    if identifier.endswith("_cls"):
        return "classification"
    if identifier.endswith("_loc"):
        return "grounding"
    if "_vqa_" in identifier:
        return "vqa"
    return "other"

def count_values(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
