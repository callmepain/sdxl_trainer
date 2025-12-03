#!/usr/bin/env python3
"""
Direkter Test: Ändern sich die Text Encoder Gewichte tatsächlich?
"""

import torch
from pathlib import Path
from safetensors.torch import load_file
import numpy as np

def compare_checkpoints(base_path, new_path, module_name="text_encoder"):
    """Vergleiche zwei Checkpoints"""

    base_file = Path(base_path) / module_name / "model.safetensors"
    new_file = Path(new_path) / module_name / "model.safetensors"

    if not base_file.exists():
        print(f"❌ Base file nicht gefunden: {base_file}")
        return None

    if not new_file.exists():
        print(f"❌ New file nicht gefunden: {new_file}")
        return None

    print(f"\n{'='*80}")
    print(f"Vergleiche {module_name}:")
    print(f"  Base: {base_path}")
    print(f"  New:  {new_path}")
    print(f"{'='*80}")

    base_state = load_file(str(base_file))
    new_state = load_file(str(new_file))

    # Finde gemeinsame Keys
    base_keys = set(base_state.keys())
    new_keys = set(new_state.keys())

    if base_keys != new_keys:
        print(f"⚠ WARNING: Keys unterscheiden sich!")
        print(f"  Nur in base: {base_keys - new_keys}")
        print(f"  Nur in new: {new_keys - base_keys}")

    common_keys = base_keys & new_keys
    print(f"\nGemeinsame Parameter: {len(common_keys)}")

    # Analysiere Unterschiede
    differences = []

    for key in sorted(common_keys):
        base_tensor = base_state[key].float()
        new_tensor = new_state[key].float()

        # Absoluter Unterschied
        abs_diff = (base_tensor - new_tensor).abs()

        # Statistiken
        mean_diff = abs_diff.mean().item()
        max_diff = abs_diff.max().item()
        std_diff = abs_diff.std().item()

        # Relative Änderung
        base_norm = base_tensor.norm().item()
        diff_norm = abs_diff.norm().item()
        rel_change = (diff_norm / base_norm * 100) if base_norm > 0 else 0

        differences.append({
            'key': key,
            'mean_diff': mean_diff,
            'max_diff': max_diff,
            'std_diff': std_diff,
            'rel_change': rel_change,
            'shape': tuple(base_tensor.shape),
        })

    # Sortiere nach größter Änderung
    differences.sort(key=lambda x: x['rel_change'], reverse=True)

    # Zeige Top 20 geänderte Parameter
    print(f"\nTop 20 am meisten geänderte Parameter:")
    print(f"{'Parameter':<60} {'Rel. Change':>12} {'Mean Diff':>12} {'Max Diff':>12}")
    print(f"{'-'*100}")

    for i, diff in enumerate(differences[:20]):
        print(f"{diff['key']:<60} {diff['rel_change']:>11.6f}% {diff['mean_diff']:>12.6e} {diff['max_diff']:>12.6e}")

    # Gesamtstatistik
    print(f"\n{'='*80}")
    print(f"ZUSAMMENFASSUNG:")
    print(f"{'='*80}")

    all_rel_changes = [d['rel_change'] for d in differences]
    all_mean_diffs = [d['mean_diff'] for d in differences]

    print(f"\nRelative Änderung (%):")
    print(f"  Mean:   {np.mean(all_rel_changes):.6f}%")
    print(f"  Median: {np.median(all_rel_changes):.6f}%")
    print(f"  Max:    {np.max(all_rel_changes):.6f}%")
    print(f"  Min:    {np.min(all_rel_changes):.6f}%")

    print(f"\nAbsolute Mean Diff:")
    print(f"  Mean:   {np.mean(all_mean_diffs):.6e}")
    print(f"  Median: {np.median(all_mean_diffs):.6e}")
    print(f"  Max:    {np.max(all_mean_diffs):.6e}")

    # Check ob sich überhaupt was geändert hat
    unchanged = sum(1 for d in differences if d['max_diff'] < 1e-10)

    print(f"\nParameter-Status:")
    print(f"  Total:     {len(differences)}")
    print(f"  Unchanged: {unchanged} ({unchanged/len(differences)*100:.1f}%)")
    print(f"  Changed:   {len(differences) - unchanged} ({(len(differences)-unchanged)/len(differences)*100:.1f}%)")

    if unchanged == len(differences):
        print(f"\n  ❌ PROBLEM: ALLE Parameter sind IDENTISCH!")
        print(f"  → Text Encoder wurde NICHT trainiert!")
        return False
    elif unchanged > len(differences) * 0.9:
        print(f"\n  ⚠ WARNING: Über 90% der Parameter unverändert!")
        print(f"  → Sehr wenig Learning")
        return False
    else:
        avg_rel_change = np.mean(all_rel_changes)
        if avg_rel_change < 0.001:
            print(f"\n  ⚠ WARNING: Durchschnittliche Änderung sehr klein ({avg_rel_change:.6f}%)")
            print(f"  → Learning Rate eventuell zu klein")
            return False
        else:
            print(f"\n  ✓ OK: Text Encoder wurde trainiert (avg change: {avg_rel_change:.6f}%)")
            return True

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python test_te_learning.py <base_checkpoint> <new_checkpoint>")
        print()
        print("Example:")
        print("  python test_te_learning.py .output/Chris_v11 .output/Chris_v13_te_aggressive")
        print()
        print("Oder für einen step checkpoint:")
        print("  python test_te_learning.py .output/Chris_v13_te_aggressive .output/Chris_v13_te_aggressive_step_250")
        sys.exit(1)

    base_path = sys.argv[1]
    new_path = sys.argv[2]

    print("="*80)
    print("TEXT ENCODER LEARNING TEST")
    print("="*80)

    # Check TE1
    te1_ok = compare_checkpoints(base_path, new_path, "text_encoder")

    # Check TE2
    te2_ok = compare_checkpoints(base_path, new_path, "text_encoder_2")

    # Optional: Check UNet zum Vergleich
    print("\n" + "="*80)
    print("Zum Vergleich: UNet (sollte sich auch ändern)")
    print("="*80)
    unet_ok = compare_checkpoints(base_path, new_path, "unet")

    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    if te1_ok and te2_ok:
        print("✓ Beide Text Encoder wurden trainiert")
    elif te2_ok and not te1_ok:
        print("⚠ Nur TE2 wurde trainiert, TE1 nicht")
    elif te1_ok and not te2_ok:
        print("⚠ Nur TE1 wurde trainiert, TE2 nicht")
    else:
        print("❌ BEIDE Text Encoder wurden NICHT trainiert!")
        print("\nMögliche Ursachen:")
        print("  1. requires_grad wurde nicht gesetzt")
        print("  2. Gradienten fließen nicht zurück")
        print("  3. Learning Rate ist zu klein")
        print("  4. Optimizer wurde nicht richtig konfiguriert")
