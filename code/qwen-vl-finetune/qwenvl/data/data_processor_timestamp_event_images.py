"""Native timestamp video plus independently encoded event images."""
import json,copy
from pathlib import Path
import torch
from PIL import Image
from . import data_processor_event_timestamp as timestamp
from event_rgb_visualization import augment_messages,processor_inputs,POLICY,EventImageConfig,validate_cached_event_meta
base=timestamp.base

def load_source(source):
    source=timestamp.with_roi_block_prompt(source)
    info=source.get('event_image_supplement')
    if not info:raise ValueError('Missing event_image_supplement; use matching preparation script')
    meta=json.loads(Path(info['metadata_path']).read_text())
    validate_cached_event_meta(meta)
    if meta['config_signature']!=info['config_signature']:raise ValueError('Event config mismatch')
    if meta.get('policy')!=info.get('policy'):raise ValueError('Event input policy mismatch')
    _,rgb_meta_path=timestamp._timestamp_cache_paths(source)
    if Path(meta['rgb_metadata_path']).resolve()!=rgb_meta_path.resolve():raise ValueError('Event images belong to a different RGB cache')
    rgb_meta=json.loads(rgb_meta_path.read_text())
    for group in meta['groups']:
        block=group['rgb_block']
        if not 1<=block<=len(rgb_meta['frames_indices'])//2:raise ValueError('Invalid RGB block reference')
        frame=rgb_meta['frames_indices'][2*(block-1)]
        if frame!=group['rgb_frame_index'] or abs(group['end']-frame/rgb_meta['fps'])>1e-9 or not 0<=group['start']<group['end']:raise ValueError('Event time is not aligned to the RGB block')
    images=[]
    for path in meta['image_paths']:
        with Image.open(path) as im:images.append(im.convert('RGB'))
    if [list(im.size) for im in images]!=[size for group in meta['groups'] for size in group['image_sizes']]:raise ValueError('Event image dimensions changed')
    messages=base._build_messages(source,Path(source.get('data_path') or base.REPO_ROOT))
    messages=augment_messages(messages,meta,images)
    frames,metadata=timestamp._load_timestamp_video(source)
    return messages,frames,metadata,images

def preprocess_qwen_visual(sources,processor):
    if len(sources)!=1:raise ValueError('One sample required')
    messages,frames,metadata,images=load_source(sources[0])
    result=processor_inputs(processor,messages,frames,metadata,images,training=True)
    result['labels']=base._make_labels(result['input_ids'])
    return result

class LazySupervisedDataset(timestamp.LazySupervisedDataset):
    def _get_item(self,sources):
        data=preprocess_qwen_visual(sources,self.processor);length=data['input_ids'].shape[-1]
        if length>self.tokenizer.model_max_length:
            raise ValueError(f"{sources[0].get('id')}: {length} exceeds context {self.tokenizer.model_max_length}; refusing RGB shrink or truncation")
        if not (data['labels']!=-100).any():raise ValueError('No supervised answer')
        seconds=[self.processor.video_processor.temporal_patch_size/self.processor.video_processor.fps]*len(data['video_grid_thw'])
        positions,_=self.get_rope_index(self.merge_size,data['input_ids'],image_grid_thw=data.get('image_grid_thw'),video_grid_thw=data['video_grid_thw'],second_per_grid_ts=seconds)
        data['position_ids']=positions;data['attention_mask']=[length]
        return data

def make_supervised_data_module(processor,data_args):
    return dict(train_dataset=LazySupervisedDataset(processor,data_args),eval_dataset=None,data_collator=base.DataCollatorForSupervisedDataset(processor.tokenizer))
