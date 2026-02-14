#!/usr/bin/env python3
"""
Test: Wie verteilen sich Gradienten durch die text_embeds Concatenation?
"""

import torch
from pathlib import Path
from config_utils import load_config
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from transformers import AutoTokenizer

cfg = load_config(Path("config.json"))
model_load_path = cfg["model"]["id"]
device = cfg["device"]
dtype = torch.bfloat16 if cfg["model"]["use_bf16"] else torch.float16

print("="*80)
print("Gradient Distribution Test")
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

# Enable gradients
unet.requires_grad_(True)
te1.requires_grad_(True)
te2.requires_grad_(True)

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

# Zero grads
te1.zero_grad()
te2.zero_grad()
unet.zero_grad()

print("\n1. FORWARD PASS:")

# Encode with grad enabled
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

print(f"   TE1 output: {prompt_embeds_1.shape}, requires_grad={prompt_embeds_1.requires_grad}")
print(f"   TE2 output: {prompt_embeds_2.shape}, requires_grad={prompt_embeds_2.requires_grad}")

if prompt_embeds_2.shape[1] != prompt_embeds_1.shape[1]:
    prompt_embeds_2 = prompt_embeds_2.expand(
        -1, prompt_embeds_1.shape[1], -1
    )

# Concatenate
text_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
print(f"   Concatenated: {text_embeds.shape}, requires_grad={text_embeds.requires_grad}")

pooled_embeds = enc_2.text_embeds.to(dtype)

# UNet forward
fake_latents = torch.randn(1, 4, 128, 128, device=device, dtype=dtype, requires_grad=True)
timestep = torch.tensor([500], device=device, dtype=torch.long)
add_time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=device, dtype=dtype)

model_pred = unet(
    fake_latents,
    timestep,
    encoder_hidden_states=text_embeds.to(dtype),
    added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids},
).sample

print(f"   UNet output: {model_pred.shape}")

# Compute loss
loss = model_pred.mean()
print(f"   Loss: {loss.item():.6f}")

print("\n2. BACKWARD PASS:")
loss.backward()

# Check gradient on concatenated tensor
if text_embeds.grad is not None:
    grad_te1_part = text_embeds.grad[:, :, :768]  # First 768 dims (TE1)
    grad_te2_part = text_embeds.grad[:, :, 768:]   # Last 1280 dims (TE2)

    te1_grad_norm = grad_te1_part.norm().item()
    te2_grad_norm = grad_te2_part.norm().item()

    print(f"   Gradient norm on TE1 part of concat: {te1_grad_norm:.6f}")
    print(f"   Gradient norm on TE2 part of concat: {te2_grad_norm:.6f}")
    print(f"   Ratio (TE2/TE1): {te2_grad_norm/te1_grad_norm:.2f}x")

# Check gradients in actual TEs
def compute_module_grad_norm(module):
    norms = []
    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        norms.append(grad.float().norm(2))
    if not norms:
        return 0.0
    return torch.norm(torch.stack(norms)).item()

te1_actual_grad = compute_module_grad_norm(te1)
te2_actual_grad = compute_module_grad_norm(te2)
unet_actual_grad = compute_module_grad_norm(unet)

print(f"\n3. ACTUAL MODULE GRADIENTS:")
print(f"   TE1 grad norm:  {te1_actual_grad:.6f}")
print(f"   TE2 grad norm:  {te2_actual_grad:.6f}")
print(f"   UNet grad norm: {unet_actual_grad:.6f}")
print(f"\n   TE2/TE1 ratio: {te2_actual_grad/te1_actual_grad if te1_actual_grad > 0 else 0:.2f}x")
print(f"   UNet/TE1 ratio: {unet_actual_grad/te1_actual_grad if te1_actual_grad > 0 else 0:.2f}x")
print(f"   UNet/TE2 ratio: {unet_actual_grad/te2_actual_grad if te2_actual_grad > 0 else 0:.2f}x")

# Analyse grad norms per layer in TEs
print(f"\n4. PER-LAYER GRADIENT ANALYSIS:")

print("\n   TE1 (CLIP ViT-L):")
te1_layer_grads = []
for name, param in te1.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        te1_layer_grads.append((name, grad_norm))

# Top 5 layers mit größten Gradienten
te1_layer_grads.sort(key=lambda x: x[1], reverse=True)
for name, grad_norm in te1_layer_grads[:5]:
    print(f"     {name[:60]:<60} {grad_norm:.6e}")

print("\n   TE2 (OpenCLIP ViT-bigG):")
te2_layer_grads = []
for name, param in te2.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        te2_layer_grads.append((name, grad_norm))

te2_layer_grads.sort(key=lambda x: x[1], reverse=True)
for name, grad_norm in te2_layer_grads[:5]:
    print(f"     {name[:60]:<60} {grad_norm:.6e}")

print("\n" + "="*80)
print("ANALYSE:")
print("="*80)

if te2_actual_grad > te1_actual_grad * 3:
    print(f"\n   ⚠ TE2 bekommt {te2_actual_grad/te1_actual_grad:.1f}x mehr Gradienten als TE1!")
    print("   \n   Mögliche Ursachen:")
    print("   1. TE2 hat mehr Parameter (695M vs 124M)")
    print("   2. TE2 hidden dim ist größer (1280 vs 768)")
    print("   3. pooled_embeds kommen nur von TE2")
    print("   4. Gradient Flow durch UNet bevorzugt TE2-Features")

print("\n" + "="*80)
print("LÖSUNG:")
print("="*80)
print("Da text_embeds 9x wichtiger sind als pooled_embeds,")
print("und TE1 37.5% der text_embeds liefert,")
print("sollte TE1 theoretisch starken Einfluss haben.")
print()
print("Wenn TE1 trotzdem schwache Gradienten hat, liegt es an:")
print("1. Unterschiedliche Hidden Dims (768 vs 1280)")
print("2. Unterschiedliche Parameteranzahl")
print("3. Mögliche Bias im UNet zu TE2-Features")
print()
print("Empfehlung: TE1 LR deutlich höher setzen (2-3x von TE2)")
print("um die strukturell schwächeren Gradienten auszugleichen!")
