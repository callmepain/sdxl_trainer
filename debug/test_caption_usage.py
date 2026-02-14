#!/usr/bin/env python3
"""
Test-Script um zu überprüfen ob Captions tatsächlich verwendet werden
und ob sie sich bei verschiedenen Inputs unterscheiden.
"""

import torch
from pathlib import Path
from config_utils import load_config
from transformers import AutoTokenizer
from dataset import SimpleCaptionDataset

def test_caption_usage():
    print("=" * 60)
    print("Caption Usage Test")
    print("=" * 60)

    cfg = load_config(Path("config.json"))
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    model_load_path = model_cfg["id"]

    # Tokenizer laden
    tokenizer_1 = AutoTokenizer.from_pretrained(
        model_load_path, subfolder="tokenizer", use_fast=False
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        model_load_path, subfolder="tokenizer_2", use_fast=False
    )

    # Dataset erstellen
    dataset = SimpleCaptionDataset(
        img_dir=data_cfg["image_dir"],
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        size=data_cfg["size"],
        caption_dropout_prob=0.0,  # Kein Dropout für den Test
        caption_shuffle_prob=0.0,  # Kein Shuffle für den Test
        bucket_config=data_cfg.get("bucket", {}),
        latent_cache_config={"enabled": False},  # Cache aus für den Test
        pixel_dtype=torch.float32,
    )

    print(f"\n1. DATASET INFO:")
    print(f"   Anzahl Samples: {len(dataset)}")
    print(f"   Image Dir: {data_cfg['image_dir']}")

    # Überprüfe die ersten paar Samples
    print(f"\n2. CAPTION CHECK (erste 5 Samples):")
    for i in range(min(5, len(dataset))):
        img_path = dataset.files[i]
        txt_path = img_path.with_suffix(".txt")

        # Caption File prüfen
        if txt_path.exists():
            caption = txt_path.read_text(encoding="utf-8").strip()
            print(f"\n   Sample {i}: {img_path.name}")
            print(f"   Caption File: ✓ vorhanden")
            print(f"   Caption Length: {len(caption)} Zeichen")
            print(f"   Caption Preview: {caption[:100]}{'...' if len(caption) > 100 else ''}")
        else:
            print(f"\n   Sample {i}: {img_path.name}")
            print(f"   Caption File: ✗ FEHLT! ({txt_path})")
            print(f"   → Leere Caption wird verwendet!")

    # Test: Werden verschiedene Captions zu verschiedenen Token-IDs?
    print(f"\n3. TOKENIZATION TEST:")

    # Erstelle Test-Captions
    test_captions = [
        "a red car",
        "a blue house",
        "a green tree",
        "",  # Leere Caption
    ]

    token_sets = []
    for caption in test_captions:
        tokens_1 = tokenizer_1(
            caption, truncation=True, max_length=77,
            padding="max_length", return_tensors="pt"
        )
        tokens_2 = tokenizer_2(
            caption, truncation=True, max_length=77,
            padding="max_length", return_tensors="pt"
        )

        # Nicht-Padding-Tokens zählen
        non_padding_1 = (tokens_1.input_ids != tokenizer_1.pad_token_id).sum().item()
        non_padding_2 = (tokens_2.input_ids != tokenizer_2.pad_token_id).sum().item()

        print(f"\n   Caption: '{caption}'")
        print(f"   Tokenizer 1: {non_padding_1} tokens (ohne padding)")
        print(f"   Tokenizer 2: {non_padding_2} tokens (ohne padding)")
        print(f"   Token IDs (first 10): {tokens_1.input_ids[0][:10].tolist()}")

        token_sets.append(tokens_1.input_ids.clone())

    # Prüfe ob verschiedene Captions zu verschiedenen Token-IDs führen
    print(f"\n4. VARIATION CHECK:")
    all_same = all(torch.equal(token_sets[0], ts) for ts in token_sets[1:])
    if all_same:
        print("   ❌ PROBLEM: Alle Captions führen zu identischen Token-IDs!")
        print("   → Das Model kann nicht zwischen verschiedenen Prompts unterscheiden!")
    else:
        print("   ✓ OK: Verschiedene Captions führen zu verschiedenen Token-IDs")

    # Test mit echten Dataset-Samples
    print(f"\n5. REAL DATASET TOKEN VARIATION:")
    if len(dataset) >= 3:
        sample_tokens = []
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            sample_tokens.append(sample["input_ids_1"].clone())

            # Zähle nicht-padding tokens
            non_pad = (sample["input_ids_1"] != tokenizer_1.pad_token_id).sum().item()
            print(f"\n   Sample {i}:")
            print(f"   Non-padding tokens: {non_pad}")
            print(f"   Token IDs (first 10): {sample['input_ids_1'][:10].tolist()}")

        # Prüfe Variation
        if len(sample_tokens) >= 2:
            all_same = all(torch.equal(sample_tokens[0], st) for st in sample_tokens[1:])
            if all_same:
                print("\n   ⚠ WARNING: Alle Samples haben identische Token-IDs!")
                print("   → Sind alle Caption-Files leer oder identisch?")
            else:
                print("\n   ✓ OK: Dataset-Samples haben unterschiedliche Token-IDs")

    # Test mit Latent Cache
    print(f"\n6. LATENT CACHE TEST:")
    if data_cfg.get("latent_cache", {}).get("enabled"):
        print("   Latent Cache ist AKTIVIERT in der Config")

        # Erstelle Dataset mit Cache
        dataset_cached = SimpleCaptionDataset(
            img_dir=data_cfg["image_dir"],
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            size=data_cfg["size"],
            caption_dropout_prob=0.0,
            caption_shuffle_prob=0.0,
            bucket_config=data_cfg.get("bucket", {}),
            latent_cache_config=data_cfg.get("latent_cache", {}),
            pixel_dtype=torch.float32,
        )

        # Simuliere Cache-Aktivierung
        if dataset_cached.latent_cache_enabled:
            # Prüfe ob ein Sample gecached ist
            if len(dataset_cached.latent_exists) > 0 and dataset_cached.latent_exists[0]:
                dataset_cached.activate_latent_cache()
                sample_cached = dataset_cached[0]

                print(f"   Sample mit Cache geladen:")
                print(f"   - Hat 'latents' key: {'latents' in sample_cached}")
                print(f"   - Hat 'pixel_values' key: {'pixel_values' in sample_cached}")
                print(f"   - Hat 'input_ids_1' key: {'input_ids_1' in sample_cached}")
                print(f"   - Non-padding tokens: {(sample_cached['input_ids_1'] != tokenizer_1.pad_token_id).sum().item()}")

                if "input_ids_1" in sample_cached:
                    print("   ✓ CAPTIONS werden auch mit Latent Cache verwendet!")
                else:
                    print("   ❌ PROBLEM: Captions fehlen bei Latent Cache!")
    else:
        print("   Latent Cache ist DEAKTIVIERT")

    print("\n" + "=" * 60)
    print("TEST ABGESCHLOSSEN")
    print("=" * 60)

    print("\n💡 NÄCHSTE SCHRITTE:")
    print("1. Stelle sicher dass alle Bilder .txt Caption-Files haben")
    print("2. Überprüfe dass die Captions nicht alle leer sind")
    print("3. Überprüfe dass verschiedene Captions verwendet werden")
    print("4. Wenn alles OK ist, könnte das Problem woanders liegen")

if __name__ == "__main__":
    test_caption_usage()
