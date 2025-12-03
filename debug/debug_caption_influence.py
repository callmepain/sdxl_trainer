#!/usr/bin/env python3
"""
Test: Welchen Einfluss haben text_embeds vs pooled_embeds auf den UNet Output?
"""

import torch
from pathlib import Path
from config_utils import load_config
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from transformers import AutoTokenizer
import numpy as np

cfg = load_config(Path("config.json"))
model_load_path = cfg["model"]["id"]
device = cfg["device"]
dtype = torch.bfloat16 if cfg["model"]["use_bf16"] else torch.float16

print("="*80)
print("Caption Influence Test: text_embeds vs pooled_embeds")
print("="*80)

# Load pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_load_path,
    torch_dtype=dtype,
    use_safetensors=True,
)
pipe.to(device)

unet = pipe.unet.to(device=device, dtype=dtype)
te1 = pipe.text_encoder.to(device=device, dtype=dtype)
te2 = pipe.text_encoder_2.to(device=device, dtype=dtype)

tokenizer_1 = AutoTokenizer.from_pretrained(
    model_load_path, subfolder="tokenizer", use_fast=False
)
tokenizer_2 = AutoTokenizer.from_pretrained(
    model_load_path, subfolder="tokenizer_2", use_fast=False
)

noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

def encode_prompt(caption):
    """Encode a caption using both TEs"""
    tokens_1 = tokenizer_1(
        caption, truncation=True, max_length=77,
        padding="max_length", return_tensors="pt"
    )
    tokens_2 = tokenizer_2(
        caption, truncation=True, max_length=77,
        padding="max_length", return_tensors="pt"
    )

    input_ids_1 = tokens_1.input_ids.to(device)
    attn_1 = tokens_1.attention_mask.to(device)
    input_ids_2 = tokens_2.input_ids.to(device)
    attn_2 = tokens_2.attention_mask.to(device)

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

    prompt_embeds_1 = enc_1.hidden_states[-2]
    prompt_embeds_2 = enc_2.hidden_states[-2]

    if prompt_embeds_2.shape[1] != prompt_embeds_1.shape[1]:
        prompt_embeds_2 = prompt_embeds_2.expand(
            -1, prompt_embeds_1.shape[1], -1
        )

    text_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    pooled_embeds = enc_2.text_embeds
    pooled_embeds_te1 = enc_1.pooler_output  # Für den Test

    return text_embeds, pooled_embeds, pooled_embeds_te1

# Test captions
captions = [
    "a red car",
    "a blue car",
    "a red house",
    "a blue house",
]

print("\n1. ENCODE CAPTIONS:")
embeddings = []
for caption in captions:
    text_emb, pooled_emb, pooled_te1 = encode_prompt(caption)
    embeddings.append((text_emb, pooled_emb, pooled_te1))
    print(f"\n   '{caption}':")
    print(f"   text_embeds shape: {text_emb.shape}")
    print(f"   pooled_embeds (TE2) shape: {pooled_emb.shape}")
    print(f"   pooled_embeds (TE1) shape: {pooled_te1.shape}")

# Vergleiche Embeddings
print("\n" + "="*80)
print("2. EMBEDDING DIFFERENCES:")
print("="*80)

def compare_embeddings(emb1, emb2, name):
    diff = (emb1 - emb2).abs().mean().item()
    return diff

print("\n   Zwischen 'red car' und 'blue car':")
text_diff_color = compare_embeddings(embeddings[0][0], embeddings[1][0], "text")
pooled_diff_color = compare_embeddings(embeddings[0][1], embeddings[1][1], "pooled")
pooled_te1_diff_color = compare_embeddings(embeddings[0][2], embeddings[1][2], "pooled_te1")

print(f"   text_embeds diff:        {text_diff_color:.6f}")
print(f"   pooled_embeds (TE2) diff: {pooled_diff_color:.6f}")
print(f"   pooled_embeds (TE1) diff: {pooled_te1_diff_color:.6f}")

print("\n   Zwischen 'red car' und 'red house':")
text_diff_object = compare_embeddings(embeddings[0][0], embeddings[2][0], "text")
pooled_diff_object = compare_embeddings(embeddings[0][1], embeddings[2][1], "pooled")
pooled_te1_diff_object = compare_embeddings(embeddings[0][2], embeddings[2][2], "pooled_te1")

print(f"   text_embeds diff:        {text_diff_object:.6f}")
print(f"   pooled_embeds (TE2) diff: {pooled_diff_object:.6f}")
print(f"   pooled_embeds (TE1) diff: {pooled_te1_diff_object:.6f}")

# Test: UNet Output mit verschiedenen Inputs
print("\n" + "="*80)
print("3. UNET SENSITIVITY TEST:")
print("="*80)

fake_latents = torch.randn(1, 4, 128, 128, device=device, dtype=dtype)
timestep = torch.tensor([500], device=device, dtype=torch.long)
add_time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=device, dtype=dtype)

print("\n   Test A: Gleiche pooled_embeds, verschiedene text_embeds")
with torch.no_grad():
    # Red car text, blue car pooled
    out1 = unet(
        fake_latents,
        timestep,
        encoder_hidden_states=embeddings[0][0].to(dtype),  # red car text
        added_cond_kwargs={"text_embeds": embeddings[1][1], "time_ids": add_time_ids},  # blue car pooled
    ).sample

    # Blue car text, blue car pooled
    out2 = unet(
        fake_latents,
        timestep,
        encoder_hidden_states=embeddings[1][0].to(dtype),  # blue car text
        added_cond_kwargs={"text_embeds": embeddings[1][1], "time_ids": add_time_ids},  # blue car pooled
    ).sample

    diff_text_only = (out1 - out2).abs().mean().item()
    print(f"   UNet output diff (nur text_embeds geändert): {diff_text_only:.6f}")

print("\n   Test B: Gleiche text_embeds, verschiedene pooled_embeds")
with torch.no_grad():
    # Red car text, red car pooled
    out1 = unet(
        fake_latents,
        timestep,
        encoder_hidden_states=embeddings[0][0].to(dtype),  # red car text
        added_cond_kwargs={"text_embeds": embeddings[0][1], "time_ids": add_time_ids},  # red car pooled
    ).sample

    # Red car text, blue car pooled
    out2 = unet(
        fake_latents,
        timestep,
        encoder_hidden_states=embeddings[0][0].to(dtype),  # red car text
        added_cond_kwargs={"text_embeds": embeddings[1][1], "time_ids": add_time_ids},  # blue car pooled
    ).sample

    diff_pooled_only = (out1 - out2).abs().mean().item()
    print(f"   UNet output diff (nur pooled_embeds geändert): {diff_pooled_only:.6f}")

print("\n   Test C: Beide geändert")
with torch.no_grad():
    # Red car
    out1 = unet(
        fake_latents,
        timestep,
        encoder_hidden_states=embeddings[0][0].to(dtype),
        added_cond_kwargs={"text_embeds": embeddings[0][1], "time_ids": add_time_ids},
    ).sample

    # Blue car
    out2 = unet(
        fake_latents,
        timestep,
        encoder_hidden_states=embeddings[1][0].to(dtype),
        added_cond_kwargs={"text_embeds": embeddings[1][1], "time_ids": add_time_ids},
    ).sample

    diff_both = (out1 - out2).abs().mean().item()
    print(f"   UNet output diff (beide geändert):          {diff_both:.6f}")

# Analyse
print("\n" + "="*80)
print("4. ANALYSE:")
print("="*80)

ratio_text_to_pooled = diff_text_only / diff_pooled_only if diff_pooled_only > 0 else 0

print(f"\n   Einfluss-Verhältnis (text_embeds / pooled_embeds): {ratio_text_to_pooled:.2f}")

if ratio_text_to_pooled > 2:
    print("   → text_embeds haben MEHR Einfluss als pooled_embeds")
    print("   → TE1 sollte durchaus wichtig sein!")
elif ratio_text_to_pooled < 0.5:
    print("   → pooled_embeds haben MEHR Einfluss als text_embeds")
    print("   → TE2 dominiert, TE1 ist weniger wichtig")
else:
    print("   → Beide haben ähnlichen Einfluss")

print(f"\n   Combined effect: {diff_both:.6f}")
print(f"   Sum of individual: {diff_text_only + diff_pooled_only:.6f}")

if diff_both > (diff_text_only + diff_pooled_only) * 0.8:
    print("   → Effekte sind additiv (beide wichtig)")
else:
    print("   → Effekte überlappen / einer dominiert")

print("\n" + "="*80)
print("FAZIT:")
print("="*80)
print("Wenn pooled_embeds deutlich mehr Einfluss haben als text_embeds,")
print("erklärt das warum TE1 (nur text_embeds) so wenig trainiert wird!")
