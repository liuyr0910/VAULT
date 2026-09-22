#!/usr/bin/env python3
import train_qwen as base_train
from qwenvl.data.data_processor_timestamp_event_images import make_supervised_data_module
base_train.make_supervised_data_module=make_supervised_data_module
if __name__=='__main__':base_train.train(attn_implementation='flash_attention_2', patch_embed_linear=True)
