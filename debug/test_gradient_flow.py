#!/usr/bin/env python3
"""
Test-Script um zu überprüfen ob Gradienten durch die Text Encoder fließen.
"""

import torch
from pathlib import Path
from config_utils import load_config
from diffusers import StableDiffusionXLPipeline

def test_gradient_flow():
    print("=" * 60)
    print("Gradient Flow Test")
    print("=" * 60)

    cfg = load_config(Path("config.json"))
    model_cfg = cfg["model"]
    device = cfg["device"]

    model_load_path = model_cfg["id"]
    use_bf16 = model_cfg["use_bf16"]
    train_text_encoder_1 = model_cfg.get("train_text_encoder_1", True)
    train_text_encoder_2 = model_cfg.get("train_text_encoder_2", False)

    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"\n1. MODEL SETUP:")
    print(f"   Model Path: {model_load_path}")
    print(f"   Device: {device}")
    print(f"   Dtype: {dtype}")
    print(f"   Train TE1: {train_text_encoder_1}")
    print(f"   Train TE2: {train_text_encoder_2}")

    # Load pipeline
    print(f"\n   Loading model...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_load_path,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.to(device)

    unet = pipe.unet.to(device=device, dtype=dtype)
    te1 = pipe.text_encoder.to(device=device, dtype=dtype)
    te2 = pipe.text_encoder_2.to(device=device, dtype=dtype)

    # Setup requires_grad
    unet.requires_grad_(True)
    te1.requires_grad_(train_text_encoder_1)
    te2.requires_grad_(train_text_encoder_2)

    print(f"\n2. REQUIRES_GRAD CHECK:")
    te1_params = sum(1 for p in te1.parameters() if p.requires_grad)
    te2_params = sum(1 for p in te2.parameters() if p.requires_grad)
    unet_params = sum(1 for p in unet.parameters() if p.requires_grad)

    print(f"   TE1 trainable params: {te1_params}")
    print(f"   TE2 trainable params: {te2_params}")
    print(f"   UNet trainable params: {unet_params}")

    if train_text_encoder_1 and te1_params == 0:
        print(f"   ❌ PROBLEM: TE1 sollte trainierbar sein, aber hat keine requires_grad Parameter!")
    if train_text_encoder_2 and te2_params == 0:
        print(f"   ❌ PROBLEM: TE2 sollte trainierbar sein, aber hat keine requires_grad Parameter!")

    # Test forward pass with two different captions
    from transformers import AutoTokenizer

    tokenizer_1 = AutoTokenizer.from_pretrained(
        model_load_path, subfolder="tokenizer", use_fast=False
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        model_load_path, subfolder="tokenizer_2", use_fast=False
    )

    test_captions = [
        "a photograph of a red sports car",
        "a photograph of a blue mountain",
    ]

    print(f"\n3. FORWARD PASS TEST:")

    embeddings_list = []

    for caption in test_captions:
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

        # Encode with grad enabled
        with torch.set_grad_enabled(train_text_encoder_1):
            enc_1 = te1(
                input_ids_1,
                attention_mask=attn_1,
                output_hidden_states=True,
                return_dict=True,
            )
        with torch.set_grad_enabled(train_text_encoder_2):
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

        print(f"\n   Caption: '{caption}'")
        print(f"   text_embeds shape: {text_embeds.shape}")
        print(f"   text_embeds requires_grad: {text_embeds.requires_grad}")
        print(f"   pooled_embeds shape: {pooled_embeds.shape}")
        print(f"   pooled_embeds requires_grad: {pooled_embeds.requires_grad}")

        embeddings_list.append((text_embeds.detach().cpu(), pooled_embeds.detach().cpu()))

    # Check if embeddings are different
    print(f"\n4. EMBEDDING VARIATION CHECK:")
    emb1_text, emb1_pooled = embeddings_list[0]
    emb2_text, emb2_pooled = embeddings_list[1]

    text_diff = (emb1_text - emb2_text).abs().mean().item()
    pooled_diff = (emb1_pooled - emb2_pooled).abs().mean().item()

    print(f"   Text embeddings mean diff: {text_diff:.6f}")
    print(f"   Pooled embeddings mean diff: {pooled_diff:.6f}")

    if text_diff < 1e-6:
        print(f"   ❌ PROBLEM: Verschiedene Captions führen zu IDENTISCHEN Text-Embeddings!")
    else:
        print(f"   ✓ OK: Verschiedene Captions führen zu unterschiedlichen Embeddings")

    # Test gradient flow
    print(f"\n5. GRADIENT FLOW TEST:")

    # Simulate one training step
    tokens_1 = tokenizer_1(
        test_captions[0], truncation=True, max_length=77,
        padding="max_length", return_tensors="pt"
    )
    tokens_2 = tokenizer_2(
        test_captions[0], truncation=True, max_length=77,
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

    # Forward pass
    with torch.set_grad_enabled(train_text_encoder_1):
        enc_1 = te1(
            input_ids_1,
            attention_mask=attn_1,
            output_hidden_states=True,
            return_dict=True,
        )
    with torch.set_grad_enabled(train_text_encoder_2):
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
    pooled_embeds = enc_2.text_embeds.to(dtype)

    # Create fake latents for UNet
    fake_latents = torch.randn(1, 4, 128, 128, device=device, dtype=dtype)
    fake_timesteps = torch.tensor([500], device=device, dtype=torch.long)

    # Create time_ids
    add_time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=device, dtype=dtype)

    # UNet forward
    model_pred = unet(
        fake_latents,
        fake_timesteps,
        encoder_hidden_states=text_embeds.to(dtype),
        added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids},
    ).sample

    # Compute fake loss
    loss = model_pred.mean()

    print(f"   Loss value: {loss.item():.6f}")
    print(f"   Loss requires_grad: {loss.requires_grad}")

    # Backward
    loss.backward()

    # Check gradients
    te1_has_grad = any(p.grad is not None and p.grad.abs().max().item() > 0 for p in te1.parameters() if p.requires_grad)
    te2_has_grad = any(p.grad is not None and p.grad.abs().max().item() > 0 for p in te2.parameters() if p.requires_grad)
    unet_has_grad = any(p.grad is not None and p.grad.abs().max().item() > 0 for p in unet.parameters() if p.requires_grad)

    print(f"\n   Gradient Check:")
    print(f"   - UNet has gradients: {unet_has_grad}")
    print(f"   - TE1 has gradients: {te1_has_grad}")
    print(f"   - TE2 has gradients: {te2_has_grad}")

    if train_text_encoder_1 and not te1_has_grad:
        print(f"   ❌ PROBLEM: TE1 sollte trainiert werden aber hat KEINE Gradienten!")
        print(f"   → Gradienten fließen nicht zurück zu TE1!")
    elif train_text_encoder_1 and te1_has_grad:
        print(f"   ✓ OK: TE1 Gradienten fließen korrekt")

    if train_text_encoder_2 and not te2_has_grad:
        print(f"   ❌ PROBLEM: TE2 sollte trainiert werden aber hat KEINE Gradienten!")
        print(f"   → Gradienten fließen nicht zurück zu TE2!")
    elif train_text_encoder_2 and te2_has_grad:
        print(f"   ✓ OK: TE2 Gradienten fließen korrekt")

    # Check grad magnitudes
    if te1_has_grad:
        te1_grad_norm = sum(p.grad.norm().item()**2 for p in te1.parameters() if p.grad is not None) ** 0.5
        print(f"\n   TE1 Gradient Norm: {te1_grad_norm:.6f}")
    if te2_has_grad:
        te2_grad_norm = sum(p.grad.norm().item()**2 for p in te2.parameters() if p.grad is not None) ** 0.5
        print(f"   TE2 Gradient Norm: {te2_grad_norm:.6f}")
    if unet_has_grad:
        unet_grad_norm = sum(p.grad.norm().item()**2 for p in unet.parameters() if p.grad is not None) ** 0.5
        print(f"   UNet Gradient Norm: {unet_grad_norm:.6f}")

    print("\n" + "=" * 60)
    print("TEST ABGESCHLOSSEN")
    print("=" * 60)

if __name__ == "__main__":
    test_gradient_flow()
