#!/usr/bin/env python3
"""Check portable paths, annotation structure, video completeness, and release hygiene."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_KEYS = {'user', 'username', 'annotator', 'email', 'timestamp', 'source_video_path', 'original_path'}


def audit(root, verify_hashes=False, private_terms=()):
    issues=[]; records=[]; split_counts={}
    for split in ('train','val','test'):
        rows=[json.loads(line) for line in (root/'data/annotations'/f'{split}.jsonl').read_text().splitlines() if line.strip()]
        split_counts[split]=len(rows); records.extend(rows)
        for row in rows:
            name=row['video_path']; path=Path(name)
            if not re.fullmatch(r'data/videos/clip_\d{6}\.mp4',name):issues.append(f'Invalid video path: {row["video_id"]}')
            if path.is_absolute() or '..' in path.parts or not (root/path).is_file():issues.append(f'Missing/nonportable video: {row["video_id"]}')
            if FORBIDDEN_KEYS & row.keys():issues.append(f'Personal metadata keys: {row["video_id"]}')
            for qa in row['vqa_pairs']:
                if qa['correct_key'] not in qa['options']:issues.append(f'Invalid VQA option: {row["video_id"]}')
    if len({r['video_id'] for r in records})!=len(records):issues.append('Duplicate IDs or overlapping splits')
    if len({r['video_path'] for r in records})!=len(records):issues.append('Duplicate video paths')
    expected={r['video_path'] for r in records}
    actual={p.relative_to(root).as_posix() for p in (root/'data/videos').glob('*')}
    if actual!=expected:issues.append('Video files differ from the annotation inventory')
    manifest=json.loads((root/'data/manifest.json').read_text())
    if {v['path'] for v in manifest['videos']}!=expected:issues.append('Manifest inventory mismatch')
    for item in manifest['videos']:
        path=root/item['path']
        if path.stat().st_size!=item['bytes']:issues.append(f'Size mismatch: {item["video_id"]}')
        if verify_hashes:
            h=hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
            if h.hexdigest()!=item['sha256']:issues.append(f'Checksum mismatch: {item["video_id"]}')
    # Inspect shipped text, including nested annotation values and the manifest.
    personal_path=re.compile(r'(?:/home/|/Users/|[A-Za-z]:[\\/](?:Users|Documents)[\\/]|/mnt/[^\s"\']+/)')
    secret=re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----')
    forbidden_names={'.git','.env','__pycache__','.ssh'}
    checked=0
    for path in root.rglob('*'):
        rel=path.relative_to(root)
        if path.is_symlink():issues.append(f'Symlink in release: {rel}')
        if path.name in forbidden_names:issues.append(f'Private/runtime artifact: {rel}')
        if not path.is_file() or path.suffix=='.mp4' or path.resolve()==Path(__file__).resolve():continue
        try:text=path.read_text()
        except UnicodeDecodeError:
            issues.append(f'Unexpected binary outside videos: {rel}');continue
        checked+=1
        if personal_path.search(text):issues.append(f'Personal absolute path in {rel}')
        if secret.search(text):issues.append(f'Credential pattern in {rel}')
        for term in private_terms:
            if re.search(r'(?<!\w)'+re.escape(term)+r'(?!\w)',text,re.I):issues.append(f'Private identifier in {rel}');break
    report={'splits':split_counts,'videos':len(records),'vqa_questions':sum(len(r['vqa_pairs']) for r in records),'text_files_checked':checked,'hashes_verified':verify_hashes,'issues':issues,'passed':not issues}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--verify-hashes',action='store_true')
    p.add_argument('--private-terms-file',type=Path,help='Optional private JSON list kept outside the release')
    p.add_argument('--report',type=Path)
    a=p.parse_args();terms=json.loads(a.private_terms_file.read_text()) if a.private_terms_file else []
    report=audit(a.root.resolve(),a.verify_hashes,terms)
    print(json.dumps(report,indent=2))
    if a.report:a.report.write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if report['passed'] else 1)

if __name__=='__main__':main()
