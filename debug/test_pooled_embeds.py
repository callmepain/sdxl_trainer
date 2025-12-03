#!/usr/bin/env python3
"""
Test: Hat TE1 überhaupt pooled embeddings?
"""

import torch
from pathlib import Path
from config_utils import load_config
from diffusers import StableDiffusionXLPipeline
from transformers import AutoTokenizer

cfg = load_config(Path("config.json"))
model_load_path = cfg["model"]["id"]
device = cfg["device"]
dtype = torch.bfloat16 if cfg["model"]["use_bf16"] else torch.float16

print("="*80)
print("TE1 Pooled Embeddings Test")
print("="*80)

# Load pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_load_path,
    torch_dtype=dtype,
    use_safetensors=True,
)
pipe.to(device)

te1 = pipe.text_encoder.to(device=device, dtype=dtype)
te2 = pipe.text_encoder_2.to(device=device, dtype=dtype)

tokenizer_1 = AutoTokenizer.from_pretrained(
    model_load_path, subfolder="tokenizer", use_fast=False
)
tokenizer_2 = AutoTokenizer.from_pretrained(
    model_load_path, subfolder="tokenizer_2", use_fast=False
)

# Test caption
test_caption = "a photograph of a red sports car"

tokens_1 = tokenizer_1(
    test_caption, truncation=True, max_length=77,
    padding="max_length", return_tensors="pt"
)
tokens_2 = tokenizer_2(
    test_caption, truncation=True, max_length=77,
    padding="max_length", return_tensors="pt"
)

input_ids_1 = tokens_1.input_ids.to(device)
attn_1 = tokens_1.attention_mask.to(device)
input_ids_2 = tokens_2.input_ids.to(device)
attn_2 = tokens_2.attention_mask.to(device)

# Encode
with torch.no_grad():
    enc_1 = te1(
        input_ids_1,
        attention_mask=attn_1,
        output_hidden_states=True,
        return_dict=True,
    )
    enc_2 = te2(
        input_ids_2,
        attention_mask=attn_2,
        output_hidden_states=True,
        return_dict=True,
    )

print("\n1. TE1 (CLIP ViT-L/14) Output:")
print(f"   Type: {type(enc_1)}")
print(f"   Keys: {enc_1.keys() if hasattr(enc_1, 'keys') else 'N/A'}")

if hasattr(enc_1, 'pooler_output'):
    print(f"   pooler_output: {enc_1.pooler_output.shape}")
    print(f"   pooler_output available: ✓")
else:
    print(f"   pooler_output available: ✗")

if hasattr(enc_1, 'text_embeds'):
    print(f"   text_embeds: {enc_1.text_embeds.shape}")
    print(f"   text_embeds available: ✓")
else:
    print(f"   text_embeds available: ✗")

if hasattr(enc_1, 'last_hidden_state'):
    print(f"   last_hidden_state: {enc_1.last_hidden_state.shape}")

print(f"   hidden_states[-2]: {enc_1.hidden_states[-2].shape}")
print(f"   hidden_states[-1]: {enc_1.hidden_states[-1].shape}")

print("\n2. TE2 (OpenCLIP ViT-bigG) Output:")
print(f"   Type: {type(enc_2)}")
print(f"   Keys: {enc_2.keys() if hasattr(enc_2, 'keys') else 'N/A'}")

if hasattr(enc_2, 'pooler_output'):
    print(f"   pooler_output: {enc_2.pooler_output.shape}")
    print(f"   pooler_output available: ✓")
else:
    print(f"   pooler_output available: ✗")

if hasattr(enc_2, 'text_embeds'):
    print(f"   text_embeds: {enc_2.text_embeds.shape}")
    print(f"   text_embeds available: ✓")
else:
    print(f"   text_embeds available: ✗")

if hasattr(enc_2, 'last_hidden_state'):
    print(f"   last_hidden_state: {enc_2.last_hidden_state.shape}")

print(f"   hidden_states[-2]: {enc_2.hidden_states[-2].shape}")
print(f"   hidden_states[-1]: {enc_2.hidden_states[-1].shape}")

print("\n" + "="*80)
print("AKTUELLER CODE:")
print("="*80)
print("text_embeds = torch.cat([enc_1.hidden_states[-2], enc_2.hidden_states[-2]], dim=-1)")
print("pooled_embeds = enc_2.text_embeds  ← NUR TE2!")
print()

# Check if we can use both
if hasattr(enc_1, 'text_embeds') or hasattr(enc_1, 'pooler_output'):
    print("="*80)
    print("MÖGLICHE VERBESSERUNG:")
    print("="*80)

    pooled_1 = getattr(enc_1, 'text_embeds', None) or getattr(enc_1, 'pooler_output', None)
    pooled_2 = enc_2.text_embeds

    if pooled_1 is not None:
        print(f"✓ TE1 hat pooled output: {pooled_1.shape}")
        print(f"✓ TE2 hat pooled output: {pooled_2.shape}")

        print("\nOptionen:")
        print("1. Concatenate beide:")
        print("   pooled_embeds = torch.cat([pooled_1, pooled_2], dim=-1)")

        print("\n2. Average beide:")
        print("   pooled_embeds = (pooled_1 + pooled_2) / 2")

        print("\n3. Weighted average (mehr Gewicht auf TE2):")
        print("   pooled_embeds = 0.3 * pooled_1 + 0.7 * pooled_2")

        print("\n4. Nur TE1 verwenden (zum Testen):")
        print("   pooled_embeds = pooled_1")

        # Test compatibility
        print("\n" + "="*80)
        print("KOMPATIBILITÄTS-TEST:")
        print("="*80)

        if pooled_1.shape == pooled_2.shape:
            print(f"✓ Shapes sind kompatibel: {pooled_1.shape} == {pooled_2.shape}")
            print("  → Average oder Weighted Average möglich")
        else:
            print(f"⚠ Shapes sind unterschiedlich: {pooled_1.shape} != {pooled_2.shape}")
            print("  → Nur Concatenation möglich")

            # Check if concat would work
            try:
                test_concat = torch.cat([pooled_1, pooled_2], dim=-1)
                print(f"  → Concatenated shape: {test_concat.shape}")
                print(f"  → Original pooled_embeds shape: {pooled_2.shape}")
                print(f"  ⚠ WARNING: UNet erwartet {pooled_2.shape}, concat gibt {test_concat.shape}")
                print(f"  → Müsste UNet-Architektur ändern!")
            except Exception as e:
                print(f"  ✗ Concat nicht möglich: {e}")
else:
    print("="*80)
    print("PROBLEM:")
    print("="*80)
    print("✗ TE1 hat KEINE pooled outputs!")
    print("  → Können nur text_embeds (hidden states) verwenden")
    print("  → Das erklärt warum TE1 so wenig Einfluss hat")
