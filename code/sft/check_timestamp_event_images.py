"""Fail-closed mixed-input preflight, including RGB tensor and label parity."""
import argparse,json,sys,hashlib,copy
from pathlib import Path
import numpy as np
import torch
from transformers import AutoProcessor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'qwen-vl-finetune'))
from qwenvl.data import data_processor_timestamp_event_images as mixed
from qwenvl.data import data_processor_event_timestamp as rgb
from event_rgb_visualization import POLICY,processor_inputs,augment_messages,validate_cached_event_meta

def audit(path,model,max_length,actual_samples=3):
 processor=AutoProcessor.from_pretrained(model,local_files_only=True);processor.tokenizer.model_max_length=max_length
 rows=json.loads(Path(path).read_text());cache={};max_len=0;longest=None;actual=[];configs=set();max_image=0
 for n,source in enumerate(rows):
  ep=source['event_image_supplement']['metadata_path'];configs.add(source['event_image_supplement']['config_signature'])
  if ep not in cache:
   meta=json.loads(Path(ep).read_text());validate_cached_event_meta(meta);rp=source['event_guided_timestamp_sampling']['metadata_path']
   if hashlib.sha256(Path(rp).read_bytes()).hexdigest()!=meta['rgb_metadata_sha256']:raise ValueError('RGB metadata changed')
   cache[ep]=meta
  meta=cache[ep]
  # Actual processor on every record is unnecessarily expensive. Expand using
  # the processor's own timestamp formatting and verify against actual below.
  upgraded=rgb.with_roi_block_prompt(source)
  from PIL import Image
  images=[Image.open(x).convert('RGB') for x in meta['image_paths']]
  messages=augment_messages(rgb.base._build_messages(upgraded,ROOT),meta,images)
  rm=json.loads(Path(source['event_guided_timestamp_sampling']['metadata_path']).read_text())
  metadata={k:rm[k] for k in ('frames_indices','fps')}
  width=(rm['width']+31)//32*32;height=(rm['height']+31)//32*32
  from event_guided_video_resolution import preserve_video_resolution,resolution_contract
  contract={'visual_tokens':len(metadata['frames_indices'])//2*width*height//1024}
  text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
  times=processor._calculate_timestamps(metadata['frames_indices'].copy(),metadata['fps'],2)
  per=width*height//1024
  expanded=''.join(f'<{t:.1f} seconds>'+processor.vision_start_token+processor.video_token*per+processor.vision_end_token for t in times)
  text=text.replace(processor.vision_start_token+processor.video_token+processor.vision_end_token,expanded,1)
  for im in images:text=text.replace(processor.image_token, '<EVENT_PLACEHOLDER>'*(im.width*im.height//1024),1)
  text=text.replace('<EVENT_PLACEHOLDER>',processor.image_token)
  ids=processor.tokenizer(text,add_special_tokens=False)['input_ids'];length=len(ids)
  if length>max_length:raise ValueError(f'{source["id"]}: {length}>{max_length}; no resizing or truncation allowed')
  if length>max_len:max_len=length;longest=source['id']
  max_image=max(max_image,sum(im.width*im.height//1024 for im in images))
  if n<actual_samples:
   real=mixed.preprocess_qwen_visual([source],processor);old=rgb.preprocess_qwen_visual([source],processor)
   assert real['input_ids'][0].tolist()==ids,'Token-only expansion does not match processor'
   for key in ('pixel_values_videos','video_grid_thw'):assert torch.equal(real[key],old[key]),key
   assert torch.equal(real['labels'][real['labels']!=-100],old['labels'][old['labels']!=-100]),'Answer labels changed'
   from qwenvl.data.rope2d import get_rope_index_3
   a,_=get_rope_index_3(2,real['input_ids'],image_grid_thw=real.get('image_grid_thw'),video_grid_thw=real['video_grid_thw'])
   b,_=get_rope_index_3(2,old['input_ids'],video_grid_thw=old['video_grid_thw'])
   assert torch.equal(a[:,0,real['input_ids'][0]==processor.video_token_id],b[:,0,old['input_ids'][0]==processor.video_token_id]),'RGB RoPE positions changed'
   # With supplement removed, the processor route is byte-for-byte baseline.
   original=rgb.with_roi_block_prompt(source);base_messages=rgb.base._build_messages(original,ROOT)
   frames,metadata=rgb._load_timestamp_video(source)
   disabled=processor_inputs(processor,base_messages,frames,metadata,[],training=True)
   assert torch.equal(disabled['input_ids'],old['input_ids'])
   actual.append(dict(id=source['id'],tokens=length,event_images=len(images),rgb_visual_tokens=contract['visual_tokens'],rgb_tensor_identical=True,rgb_rope_identical=True,answer_labels_identical=True,disabled_input_ids_identical=True))
  if n%500==0:print(f'checked {n+1}/{len(rows)}',flush=True)
 if len(configs)!=1:raise ValueError('Mixed event render configs in dataset')
 result=dict(policy=next(iter(cache.values()))['policy'],records=len(rows),videos=len(cache),max_sequence_tokens=max_len,longest_id=longest,max_event_tokens=max_image,model_max_length=max_length,actual_processor_checks=actual,config_signature=next(iter(configs)),event_config=next(iter(cache.values()))['config'])
 print(json.dumps(result,indent=2));return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-path',required=True);p.add_argument('--model',required=True);p.add_argument('--model-max-length',type=int,default=6144);p.add_argument('--actual-samples',type=int,default=3);p.add_argument('--output-dir');a=p.parse_args()
 result=audit(a.data_path,a.model,a.model_max_length,a.actual_samples)
 if a.output_dir:
  out=Path(a.output_dir);marker=out/'timestamp_event_images_policy.json'
  if list(out.glob('checkpoint-*')) and (not marker.exists() or json.loads(marker.read_text()).get('config_signature')!=result['config_signature']):raise ValueError('Checkpoint event config mismatch')
  out.mkdir(parents=True,exist_ok=True);marker.write_text(json.dumps(result,indent=2))
 Path(a.data_path).with_suffix('.validation.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
