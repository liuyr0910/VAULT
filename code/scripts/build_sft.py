#!/usr/bin/env python3
"""Build four-task instruction records from portable benchmark annotations."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'inf_script'))
spec = importlib.util.spec_from_file_location('vault_inference', ROOT / 'inf_script/infer-all-timestamp-event-images.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def build_records(rows):
    records = []
    for row in rows:
        identifier = row['video_id']
        video = Path(row['video_path'])
        if video.is_absolute() or '..' in video.parts:
            raise ValueError('Benchmark video paths must be relative to the release root')
        tasks = [('d', 'desc', row['description'], None),
                 ('c', 'cls', ', '.join(row['category']), None),
                 ('t', 'loc', ', '.join(f'{a:g} - {b:g}' for a, b in row['temporal_segments']), None)]
        for i, qa in enumerate(row['vqa_pairs']):
            if qa['correct_key'] not in qa['options']:
                raise ValueError(f'{identifier}: unknown correct option')
            tasks.append(('vqa', f'vqa_{i}', qa['correct_key'], qa))
        for task, suffix, answer, qa in tasks:
            records.append({'id': f'{identifier}_{suffix}', 'video': video.as_posix(),
                            'conversations': [
                                {'from': 'human', 'value': f'<video>{video.as_posix()}</video>\n' + module.PromptFactory.task_prompt(task, row, qa)},
                                {'from': 'gpt', 'value': answer}]})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=ROOT/'data/annotations/train.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT/'sft/data/all_tasks_mixed.json')
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.annotations.read_text().splitlines() if line.strip()]
    records = build_records(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if json.loads(args.output.read_text()) == records:
            print(f'Existing {len(records)} records match; reused.')
            return
        raise FileExistsError('Output differs from requested annotations; choose a new output path')
    with args.output.open('x') as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(f'Created {len(records)} instruction records from {len(rows)} videos.')

if __name__ == '__main__':
    main()
