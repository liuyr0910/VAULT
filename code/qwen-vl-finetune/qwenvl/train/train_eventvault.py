# Adapted for EventVAULT training and multimodal data inputs.
"""Train the EventVAULT LoRA model."""

import train_qwen

if __name__ == "__main__":
    train_qwen.train(attn_implementation="flash_attention_2", patch_embed_linear=True)
