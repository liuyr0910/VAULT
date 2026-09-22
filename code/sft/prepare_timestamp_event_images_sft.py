#!/usr/bin/env python3
"""Add cached event PNGs to existing RGB timestamp records, preserving all RGB."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'inf_script'))
from event_rgb_visualization import *

def prepare_one(args):
 path,cache_root,config=args;c=EventImageConfig(**config);m=json.loads(Path(path).read_text())
 event_fingerprint=fingerprint(m['event_path']) if c.max_groups else None
 key=hashlib.sha256((str(Path(path).resolve())+c.signature()+json.dumps(event_fingerprint,sort_keys=True)).encode()).hexdigest()[:20]
 folder=Path(cache_root)/key;meta_path=folder/'event.json';rgb_sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
 if meta_path.exists():
  meta=json.loads(meta_path.read_text())
  if meta['rgb_metadata_sha256']!=rgb_sha or any(not Path(x).is_file() for x in meta['image_paths']):raise ValueError('Invalid event cache; use a new cache directory')
 else:
  images,meta=build_bundle(m['event_path'],m['rgb_samples'],m['fps'],m['duration'],c)
  folder.mkdir(parents=True,exist_ok=True);paths=[]
  for i,im in enumerate(images):
   target=folder/f'{i:02d}.png';im.save(target);paths.append(str(target.resolve()))
  meta.update(image_paths=paths,rgb_metadata_sha256=rgb_sha,rgb_metadata_path=str(Path(path).resolve()))
  tmp=meta_path.with_suffix('.tmp');tmp.write_text(json.dumps(meta,indent=2));tmp.replace(meta_path)
 return path,str(meta_path.resolve())

def reuse_existing(output, input_path, source_rows, selected_paths, config):
 """Reuse only a complete export matching the requested source and rendering."""
 manifest_path=output.with_suffix('.manifest.json')
 try:
  manifest=json.loads(manifest_path.read_text())
  existing=json.loads(output.read_text())
  expected=[row for row in source_rows if row['event_guided_timestamp_sampling']['metadata_path'] in selected_paths]
  if manifest['source_sha256']!=hashlib.sha256(input_path.read_bytes()).hexdigest():
   raise ValueError('source dataset has changed')
  if manifest['config_signature']!=config.signature():
   raise ValueError('requested event configuration differs from the existing export')
  if manifest['output_sha256']!=hashlib.sha256(output.read_bytes()).hexdigest():
   raise ValueError('existing dataset is incomplete or was modified')
  if len(existing)!=len(expected) or manifest['records']!=len(expected):
   raise ValueError('requested video subset differs from the existing export')
  checked=set()
  print(f'Output exists; validating {len(existing)} records and cached event images before reuse...',flush=True)
  for original,record in zip(expected,existing):
   record=dict(record);info=record.pop('event_image_supplement')
   if record!=original or info['config_signature']!=config.signature() or info['policy']!=POLICY:
    raise ValueError('existing records do not match the source or rendering policy')
   path=Path(info['metadata_path'])
   if path in checked:continue
   meta=json.loads(path.read_text());rgb_path=Path(original['event_guided_timestamp_sampling']['metadata_path'])
   if Path(meta['rgb_metadata_path']).resolve()!=rgb_path.resolve() or meta['rgb_metadata_sha256']!=hashlib.sha256(rgb_path.read_bytes()).hexdigest():
    raise ValueError(f'RGB cache metadata changed: {rgb_path}')
   if meta['config_signature']!=config.signature() or meta['policy']!=POLICY:
    raise ValueError(f'event cache configuration differs: {path}')
   if config.max_groups:
    rgb_meta=json.loads(rgb_path.read_text())
    if meta['event_fingerprint']!=fingerprint(rgb_meta['event_path']):
     raise ValueError(f'event H5 changed: {rgb_meta["event_path"]}')
   sizes=[size for group in meta['groups'] for size in group['image_sizes']]
   if len(meta['image_paths'])!=len(sizes):raise ValueError(f'event image count mismatch: {path}')
   for image_path,size in zip(meta['image_paths'],sizes):
    with Image.open(image_path) as image:
     if image.format!='PNG' or image.mode!='RGB' or list(image.size)!=size:
      raise ValueError(f'event image format/size changed: {image_path}')
     image.verify()
   checked.add(path)
   if len(checked)%100==0:print(f'Validated {len(checked)} event caches',flush=True)
 except (OSError, ValueError, KeyError, TypeError) as exc:
  raise ValueError(f'Cannot reuse existing output {output}: {exc}. '
                   'Use --output-json with a new filename for a different configuration or rebuild; existing data were not overwritten.') from exc
 print(f'Existing dataset validated and reused: {output}\n'
       f'{len(existing)} records, {len(checked)} video caches; no regeneration needed.',flush=True)
 return manifest


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-json',default=str(ROOT/'sft/data/all_tasks_mixed_event_guided_timestamp.json'));p.add_argument('--output-json',default=str(ROOT/'sft/data/all_tasks_mixed_timestamp_event_images.json'));p.add_argument('--config',default=str(ROOT/'sft/event_rgb/configs/adaptive_gray_pair.json'));p.add_argument('--cache-root',default=str(ROOT/'sft/cache/timestamp_event_images'));p.add_argument('--workers',type=int,default=2);p.add_argument('--limit-videos',type=int,default=0);a=p.parse_args()
 output=Path(a.output_json)
 if a.workers<1 or a.limit_videos<0:raise ValueError("workers must be positive and limit-videos nonnegative")
 rows=json.loads(Path(a.input_json).read_text());c=load_config(a.config);unique=list(dict.fromkeys(r['event_guided_timestamp_sampling']['metadata_path'] for r in rows))
 manifest=json.loads(Path(a.input_json).with_suffix('.manifest.json').read_text())
 c=bind_sampler_config(c,EventSamplerConfig(**manifest['event_sampler_config']))
 if a.limit_videos:unique=unique[:a.limit_videos]
 if output.exists():
  reuse_existing(output,Path(a.input_json),rows,set(unique),c)
  return
 jobs=[(s,a.cache_root,asdict(c)) for s in unique];mapping={}
 if a.workers==1:
  for job in jobs:
   path,target=prepare_one(job);mapping[path]=target;print(f'{len(mapping)}/{len(jobs)} event caches',flush=True)
 else:
  with ProcessPoolExecutor(max_workers=a.workers) as pool:
   for path,target in pool.map(prepare_one,jobs):mapping[path]=target;print(f'{len(mapping)}/{len(jobs)} event caches',flush=True)
 result=[]
 for row in rows:
  path=row['event_guided_timestamp_sampling']['metadata_path']
  if path not in mapping:continue
  r=copy.deepcopy(row);r['event_image_supplement']={'metadata_path':mapping[path],'policy':POLICY,'config_signature':c.signature()};result.append(r)
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
 report=dict(records=len(result),videos=len(mapping),config=asdict(c),config_signature=c.signature(),source_sha256=hashlib.sha256(Path(a.input_json).read_bytes()).hexdigest(),output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),baseline_fields_preserved=True)
 output.with_suffix('.manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
