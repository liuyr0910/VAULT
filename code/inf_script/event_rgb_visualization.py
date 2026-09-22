"""Auditable event images supplementing (never replacing) timestamp RGB.

H5 contract: v2e events=[timestamp_us,x,y,polarity], source-aligned 240x180.
Window candidates are source-frame periods, not assumed sensor microseconds.
"""
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from PIL import Image
from event_guided_sampling import _import_h5py, EventSamplerConfig, analyze_event_h5

POLICY = 'timestamp_event_images_v3_reliable_segments'
SCORE_FIELDS = ('analysis_bins', 'spatial_grid_width', 'spatial_grid_height',
                'activity_threshold', 'min_temporal_contrast', 'merge_gap_bins')

@dataclass(frozen=True)
class EventImageConfig:
    max_groups: int = 3
    filter_inactive_anchors: bool = True
    one_anchor_per_segment: bool = True
    balanced_event_note: bool = False
    score_config: dict = field(default_factory=lambda: {k: getattr(EventSamplerConfig(), k) for k in SCORE_FIELDS})
    frame_windows: tuple = (3, 6, 12, 24, 48)
    target_events_per_pixel: float = 1.0  # train-only audit; see sft/event_rgb/audit/report.json
    representation: str = 'gray_pair'
    sensor_width: int = 240
    sensor_height: int = 180
    timestamp_divisor: float = 1_000_000.0
    max_read_events: int = 4_000_000
    def __post_init__(self):
        if self.max_groups < 0 or self.target_events_per_pixel <= 0:
            raise ValueError('Invalid group count or event target')
        if not self.frame_windows or any(x < 3 for x in self.frame_windows) or sorted(set(self.frame_windows)) != list(self.frame_windows):
            raise ValueError('frame_windows must be increasing unique values >=3')
        if self.representation not in ('gray_pair', 'red_blue'):
            raise ValueError('Unknown event representation')
        if min(self.sensor_width,self.sensor_height,self.timestamp_divisor,self.max_read_events)<=0:
            raise ValueError('Invalid sensor configuration')
    def signature(self):
        return hashlib.sha256(json.dumps(dict(policy=POLICY,config=asdict(self)),sort_keys=True).encode()).hexdigest()[:16]

def load_config(path):
    data=json.loads(Path(path).read_text());data=data.get('config',data)
    return EventImageConfig(**data)

def validate_cached_event_meta(meta):
    """Read existing v2/v3 training caches without migrating their inputs."""
    policy=meta.get('policy')
    if policy not in ('timestamp_event_images_v2_score_ranked',POLICY):
        raise ValueError('Unsupported cached event input policy')
    config=meta['config']
    EventImageConfig(**config)  # Validate values, but hash the stored schema.
    if policy=='timestamp_event_images_v2_score_ranked' and any(
            key in config for key in ('filter_inactive_anchors','one_anchor_per_segment','balanced_event_note')):
        raise ValueError('v2 cache contains v3 switches')
    expected=hashlib.sha256(json.dumps(dict(policy=policy,config=config),sort_keys=True).encode()).hexdigest()[:16]
    if expected!=meta.get('config_signature'):
        raise ValueError('Cached event config signature mismatch')


def fingerprint(path):
    p=Path(path).resolve();s=p.stat()
    return dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns)

def bind_sampler_config(config, sampler):
    """Carry the RGB sampler's exact scoring settings into cache provenance."""
    for key in ('sensor_width', 'sensor_height', 'timestamp_divisor'):
        if getattr(config, key) != getattr(sampler, key):
            raise ValueError(f'RGB/event sampler {key} mismatch')
    return replace(config, score_config={k: getattr(sampler, k) for k in SCORE_FIELDS})

def anchor_indices(samples, max_groups, scores, duration, fps, *, min_score=None, segments=None):
    """Top-scoring distinct source-time bins; retain chronological input order.

    All RGB roles are eligible. Ties prefer the earlier frame, then block index.
    Optional filtering reuses the original activity threshold. When segments
    are supplied, select at most one anchor per original active segment.
    """
    if not max_groups or not samples:return []
    if duration <= 0 or fps <= 0 or not len(scores):raise ValueError('Invalid anchor scoring clock')
    pool=[]
    for i,sample in enumerate(samples):
        time=int(sample['frame_index'])/fps
        if not 0 < time <= duration:continue
        event_bin=min(len(scores)-1,int(time*len(scores)/duration))
        score=float(scores[event_bin])
        if min_score is not None and (score<=0 or score<min_score):continue
        pool.append((i,event_bin,time,score))
    pool.sort(key=lambda x:(-x[3],x[2],x[0]))
    selected=[];seen=set();seen_segments=set()
    for i,event_bin,time,score in pool:
        if event_bin in seen:continue
        segment_id=None
        if segments is not None:
            segment_id=next((j for j,seg in enumerate(segments) if seg.start_bin<=event_bin<=seg.end_bin),None)
            if segment_id is None or segment_id in seen_segments:continue
        selected.append(i);seen.add(event_bin)
        if segment_id is not None:seen_segments.add(segment_id)
        if len(selected)==max_groups:break
    return sorted(selected,key=lambda i:(samples[i]['frame_index'],i))

class EventClockError(ValueError):
    pass

class EventReader:
    def __init__(self,path,config,duration):
        self.config=config;self.handle=_import_h5py().File(path,'r');self.events=self.handle['events']
        if self.events.ndim!=2 or self.events.shape[1]!=4:
            self.close();raise ValueError('Expected v2e Nx4 H5 events')
        if len(self.events):
            last=float(self.events[-1,0])/config.timestamp_divisor
            if last>duration+max(1.,duration*.02):
                self.close();raise EventClockError(f'Event end {last:.6f}s exceeds RGB duration {duration:.6f}s; verify source clock')
    def close(self):self.handle.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def bound(self,time):
        lo,hi=0,len(self.events);value=time*self.config.timestamp_divisor
        while lo<hi:
            mid=(lo+hi)//2
            if float(self.events[mid,0])<value:lo=mid+1
            else:hi=mid
        return lo
    def chunks(self,start,end,roi):
        lo,hi=self.bound(start),self.bound(end);last=None
        c=self.config;l,t,r,b=roi
        # Bounded streaming, preserving EVERY event even in very active clips.
        for offset in range(lo,hi,c.max_read_events):
            e=np.asarray(self.events[offset:min(hi,offset+c.max_read_events)],dtype=np.float64)
            if len(e) and (np.any(np.diff(e[:,0])<0) or (last is not None and e[0,0]<last) or not np.isin(e[:,3],[0,1,-1]).all()):
                raise ValueError('Unsorted timestamps or unknown polarity')
            if len(e):last=e[-1,0]
            if len(e) and (np.any(e[:,1:3]<0) or np.any(e[:,1]>=c.sensor_width) or np.any(e[:,2]>=c.sensor_height)):
                raise ValueError('Event coordinates exceed configured sensor dimensions')
            mask=(e[:,1]>=l)&(e[:,1]<r)&(e[:,2]>=t)&(e[:,2]<b)
            e=e[mask];e[:,0]/=c.timestamp_divisor;e[:,1]-=l;e[:,2]-=t
            yield e
    def read(self,start,end,roi):
        pieces=list(self.chunks(start,end,roi))
        return np.concatenate(pieces) if pieces else np.empty((0,4))

def sensor_roi(sample,c):
    box=sample.get('roi') if sample.get('pixel_box') is not None else None
    box=box or [0,0,1,1]
    l,t,r,b=box
    if not (0<=l<r<=1 and 0<=t<b<=1):raise ValueError('Invalid normalized ROI')
    return [math.floor(l*c.sensor_width),math.floor(t*c.sensor_height),math.ceil(r*c.sensor_width),math.ceil(b*c.sensor_height)]

def histogram(e,start,end,shape):
    h,w=shape;out=np.zeros((3,2,h,w),np.float32)
    if len(e):
        bins=np.minimum(2,((e[:,0]-start)/(end-start)*3).astype(int))
        if np.any(bins<0):raise ValueError('Event before window')
        np.add.at(out,(bins,(e[:,3]>0).astype(int),e[:,2].astype(int),e[:,1].astype(int)),1)
    return out

def stats(hist):
    counts=hist.sum(axis=(1,2,3));occupancy=(hist.sum(axis=1)>0).mean(axis=(1,2))
    return dict(counts=counts.astype(int).tolist(),occupancy=occupancy.tolist(),empty_bins=int((counts==0).sum()),events_per_pixel=float(counts.sum()/np.prod(hist.shape[-2:])))

def candidates(reader,sample,fps,c):
    end=int(sample['frame_index'])/fps;roi=sensor_roi(sample,c);h,w=roi[3]-roi[1],roi[2]-roi[0]
    if end<=0:return []
    starts=[max(0,end-n/fps) for n in c.frame_windows]
    histograms=[np.zeros((3,2,h,w),np.float32) for _ in starts]
    for events in reader.chunks(min(starts),end,roi):
        for start,hist in zip(starts,histograms):
            part=events[events[:,0]>=start]
            hist+=histogram(part,start,end,(h,w))
    return [(dict(frame_periods=n,start=start,end=end,roi_sensor=roi,**stats(hist)),hist)
            for n,start,hist in zip(c.frame_windows,starts,histograms)]

def choose(options,c):
    # Shortest window reaching an area-normalized event target with 3 nonempty
    # chronological bins. Sparse windows remain explicitly marked, not fabricated.
    for meta,hist in options:
        if meta['events_per_pixel']>=c.target_events_per_pixel and not meta['empty_bins']:
            return meta,hist
    return options[-1]

def render(hist,representation):
    # One shared robust scale for all three bins and both polarities; no
    # per-bin contrast equalization, denoising, polarity subtraction or RGB blend.
    positive=hist[hist>0];scale=max(1.,float(np.percentile(positive,99))) if len(positive) else 1.
    v=np.minimum(1,np.log1p(hist)/np.log1p(scale));images=[]
    for plane in v:
        if representation=='gray_pair':
            neg,pos=plane;h,w=pos.shape
            gray=np.full((h,2*w+16),255,np.uint8)
            gray[:,:w]=np.rint(255*(1-pos)).astype(np.uint8)
            gray[:,w+16:]=np.rint(255*(1-neg)).astype(np.uint8)
            rgb=np.repeat(gray[:,:,None],3,axis=2);bg=255
        else:
            h,w=plane.shape[1:];rgb=np.zeros((h,w,3),np.uint8)
            rgb[:,:,0]=np.rint(plane[1]*255).astype(np.uint8);rgb[:,:,2]=np.rint(plane[0]*255).astype(np.uint8);bg=0
        h,w=rgb.shape[:2]
        rgb=np.pad(rgb,((0,(-h)%32),(0,(-w)%32),(0,0)),constant_values=bg)
        images.append(Image.fromarray(rgb))
    return images,scale

def build_bundle(event_path,samples,fps,duration,c):
    meta=dict(policy=POLICY,config=asdict(c),config_signature=c.signature(),groups=[],status='disabled' if c.max_groups==0 else 'missing_event')
    if not c.max_groups:return [],meta
    if not event_path or not Path(event_path).is_file():raise FileNotFoundError(f'Event H5 required: {event_path}')
    images=[]
    try:
        reader=EventReader(event_path,c,duration)
    except EventClockError as exc:
        meta.update(status='invalid_clock_rgb_only',error=str(exc),event_fingerprint=fingerprint(event_path),image_count=0,visual_tokens=0)
        return [],meta
    with reader:
        sampler=EventSamplerConfig(**c.score_config,sensor_width=c.sensor_width,
            sensor_height=c.sensor_height,timestamp_divisor=c.timestamp_divisor)
        analysis=analyze_event_h5(event_path,duration,sampler)
        meta['anchor_selection']=dict(method='filtered_score_segment_selection',
            filter_inactive_anchors=c.filter_inactive_anchors,one_anchor_per_segment=c.one_anchor_per_segment,
            scores=analysis.scores.tolist(),analysis=analysis.summary(),score_config=c.score_config)
        for i in anchor_indices(samples,c.max_groups,analysis.scores,duration,fps,
                min_score=sampler.activity_threshold if c.filter_inactive_anchors else None,
                segments=analysis.segments if c.one_anchor_per_segment else None):
            options=candidates(reader,samples[i],fps,c)
            if not options:continue
            chosen,hist=choose(options,c);group_images,scale=render(hist,c.representation)
            group={**chosen,'rgb_block':i+1,'rgb_frame_index':int(samples[i]['frame_index']),'roi_normalized':samples[i].get('roi') if samples[i].get('pixel_box') is not None else None,'scale_count_p99':scale,'image_sizes':[list(im.size) for im in group_images],'target_reached':chosen['events_per_pixel']>=c.target_events_per_pixel,'candidate_statistics':[m for m,_ in options]}
            event_bin=min(len(analysis.scores)-1,int(int(samples[i]['frame_index'])/fps*len(analysis.scores)/duration))
            group.update(anchor_event_bin=event_bin,anchor_score=float(analysis.scores[event_bin]),
                anchor_segment=next((j for j,seg in enumerate(analysis.segments) if seg.start_bin<=event_bin<=seg.end_bin),None))
            meta['groups'].append(group);images.extend(group_images)
    meta.update(status='ok' if images else 'no_eligible_event_anchor',event_fingerprint=fingerprint(event_path),image_count=len(images),visual_tokens=sum(im.width*im.height//1024 for im in images))
    return images,meta

def event_content(meta,images):
    if len(images)!=3*len(meta['groups']):raise ValueError('Event image/group count mismatch')
    if not images:return []
    rep=meta['config']['representation']
    legend=('Each image has two grayscale panels: LEFT=positive brightness-change events, RIGHT=negative events. Darker means more events; white means no events. Panels show the SAME region, not different places.' if rep=='gray_pair' else 'Red=positive brightness-change events, blue=negative events; brighter means more events; black means no events.')
    content=[dict(type='text',text='Supplementary event observations. '+legend+' These are event count maps, not natural photographs or object colors. Brightness change can also come from camera motion or lighting; it does not by itself establish object motion or anomaly. Each group uses a shared intensity scale. Read actual intervals; do not assume equal time gaps between groups. RGB remains the source of appearance and scene context.')]
    if meta['config'].get('balanced_event_note',False):
        content[0]['text'] += (' These maps cover only the listed local time intervals and regions, not the whole video.'
            ' Missing, weak, or ambiguous event evidence does not establish that the video is normal and does not negate an anomaly visible in RGB.'
            ' Determine the answer from all RGB observations; use event maps only as additional evidence when interpretable.'
            ' Event window boundaries are observation boundaries, not anomaly start or end labels.')
    k=0
    for g in meta['groups']:
        region=g['roi_normalized'] or [0,0,1,1];edges=np.linspace(g['start'],g['end'],4)
        for j,name in enumerate(('early','middle','late')):
            text=f"Event for RGB block {g['rgb_block']}, region={','.join(f'{x:.4f}' for x in region)}, {name} interval=[{edges[j]:.6f},{edges[j+1]:.6f}) seconds; events={g['counts'][j]}."
            content.extend([dict(type='text',text=text),dict(type='image',image=images[k])]);k+=1
    return content

def augment_messages(messages,meta,images):
    """Insert after existing video/sampling+ROI note, before task; preserve text."""
    import copy
    from event_guided_video_resolution import ROI_BLOCK_MAP_HEADER
    result=copy.deepcopy(messages);extra=event_content(meta,images)
    if not extra:return result
    users=[m for m in result if m['role']=='user']
    if len(users)!=1:raise ValueError('Expected one user turn')
    content=users[0]['content'];new=[];inserted=False
    for item in content:
        if item['type']=='text' and ROI_BLOCK_MAP_HEADER in item['text']:
            text=item['text'];idx=text.find('\n',text.index(ROI_BLOCK_MAP_HEADER))
            if idx<0:idx=len(text)
            else:idx+=1
            new.append(dict(type='text',text=text[:idx]));new.extend(extra)
            if text[idx:]:new.append(dict(type='text',text=text[idx:]))
            inserted=True
        else:new.append(item)
    if not inserted:raise ValueError('Missing RGB ROI block note; refusing ambiguous insertion')
    users[0]['content']=new
    return result

def processor_inputs(processor,messages,frames,metadata,images,training=False):
    from event_guided_video_resolution import preserve_video_resolution,resolution_contract
    frames,metadata=preserve_video_resolution(frames,metadata)
    # vLLM accepts this control flag in its video tuple, but HF VideoMetadata
    # does not. Match vLLM's adapter; keep the caller's metadata untouched.
    metadata=dict(metadata)
    metadata.pop('do_sample_frames',None)
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=not training)
    result=processor(text=[text],videos=[frames],video_metadata=[metadata],images=images or None,do_resize=False,do_sample_frames=False,return_tensors='pt')
    if result['video_grid_thw'].tolist()!=[resolution_contract(frames)['video_grid_thw']]:raise ValueError('RGB grid changed')
    if images:
        expected=[[1,im.height//16,im.width//16] for im in images]
        if result['image_grid_thw'].tolist()!=expected:raise ValueError('Event images resized unexpectedly')
        if int((result['input_ids']==processor.image_token_id).sum())!=sum(im.width*im.height//1024 for im in images):raise ValueError('Image token mismatch')
    return result
