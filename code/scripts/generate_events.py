"""Generate v2e events from anonymous videos; use a separate v2e environment."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument(
        "--v2e",
        type=Path,
        required=True,
        help="Path to an installed v2e.py entry point",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable in the v2e environment",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    rows = [
        json.loads(line)
        for line in (ROOT / "data/annotations" / f"{args.split}.jsonl")
        .read_text()
        .splitlines()
    ]
    if args.limit:
        rows = rows[: args.limit]
    for row in rows:
        source = ROOT / row["video_path"]
        folder = ROOT / "data/events" / args.split / row["video_id"]
        target = folder / (row["video_id"] + ".h5")
        if target.exists():
            print(f"Reuse {row['video_id']}")
            continue
        info = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    str(source),
                ]
            )
        )
        command = [
            args.python,
            str(args.v2e.resolve()),
            "-i",
            str(source),
            "-o",
            str(folder),
            "--overwrite",
            "--disable_slomo",
            "--no_preview",
            "--skip_video_output",
            "--dvs240",
            "--dvs_params",
            "clean",
            "--start_time",
            "0",
            "--stop_time",
            info["format"]["duration"],
            "--dvs_h5",
            target.name,
            "--vid_orig",
            "None",
            "--vid_slomo",
            "None",
            "--dvs_exposure",
            "duration",
            "0.033333",
        ]
        if args.dry_run:
            print(" ".join(command))
            continue
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=args.v2e.resolve().parent, check=True)
        print(f"Generated {row['video_id']}")


if __name__ == "__main__":
    main()
