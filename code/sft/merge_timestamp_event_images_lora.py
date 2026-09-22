#!/usr/bin/env python3
"""CPU merge of timestamp event-image LoRA; never overwrite an existing model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-model', type=Path, required=True)
    a = p.parse_args()
    adapter, output, base = a.adapter.resolve(), a.output.resolve(), a.base_model.resolve()
    if output.exists():
        raise FileExistsError(output)
    policy_path = adapter/'timestamp_event_images_policy.json'
    if not policy_path.exists():policy_path=adapter.parent/'timestamp_event_images_policy.json'
    policy=json.loads(policy_path.read_text())
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inf_script'))
    from event_rgb_visualization import POLICY, EventImageConfig
    if policy.get('policy') != POLICY:
        raise ValueError('Adapter must be trained with this release event policy')
    if EventImageConfig(**policy['event_config']).signature() != policy['config_signature']:
        raise ValueError('Adapter event configuration signature mismatch')
    config = json.loads((adapter/'adapter_config.json').read_text())
    if Path(config['base_model_name_or_path']).resolve() != base:
        raise ValueError('Adapter base model differs from --base-model')
    # Correct malformed inherited CUDA_HOME before importing torch/DeepSpeed.
    import os
    cuda_home = os.environ.get('CUDA_HOME', '')
    if ':' in cuda_home:
        candidates = [x for x in cuda_home.split(':') if x and (Path(x)/'bin/nvcc').is_file()]
        if candidates:
            os.environ['CUDA_HOME'] = candidates[0]
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(base), dtype=torch.bfloat16, device_map='cpu', local_files_only=True)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload(safe_merge=True)
    model.config.use_cache = True
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.event-images-merge-', dir=output.parent))
    try:
        model.save_pretrained(stage, safe_serialization=True)
        AutoProcessor.from_pretrained(base, local_files_only=True).save_pretrained(stage)
        digest = hashlib.sha256()
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            with (adapter/name).open('rb') as f:
                for block in iter(lambda: f.read(1024*1024), b''):
                    digest.update(block)
        (stage/'timestamp_event_images_merge.json').write_text(json.dumps(dict(adapter=str(adapter), base_model=str(base), adapter_sha256=digest.hexdigest()), indent=2))
        (stage/'timestamp_event_images_policy.json').write_text(json.dumps(policy,indent=2))
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(output)


if __name__ == '__main__':
    main()
