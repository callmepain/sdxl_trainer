#!/usr/bin/env python3
"""
Analysiere TensorBoard Logs um Gradient-Normen zu überprüfen
"""

from pathlib import Path
import sys

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("ERROR: tensorboard nicht installiert!")
    print("Installiere mit: pip install tensorboard")
    sys.exit(1)

def analyze_tensorboard_logs(run_name="Chris_v12_te_fixed"):
    log_dir = Path("logs/tensorboard") / run_name

    if not log_dir.exists():
        print(f"ERROR: Log-Verzeichnis nicht gefunden: {log_dir}")
        return

    print("=" * 80)
    print(f"TensorBoard Log Analyse: {run_name}")
    print("=" * 80)

    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()

    scalar_tags = ea.Tags()['scalars']

    print(f"\n1. VERFÜGBARE METRIKEN ({len(scalar_tags)} gesamt):")

    # Gruppiere nach Kategorie
    categories = {}
    for tag in scalar_tags:
        category = tag.split('/')[0] if '/' in tag else 'other'
        categories.setdefault(category, []).append(tag)

    for category in sorted(categories.keys()):
        tags = categories[category]
        print(f"\n   {category}/ ({len(tags)} Metriken):")
        for tag in sorted(tags)[:10]:  # Zeige max 10
            print(f"     - {tag}")
        if len(tags) > 10:
            print(f"     ... und {len(tags) - 10} weitere")

    # Analysiere Gradient-Normen
    print("\n" + "=" * 80)
    print("2. GRADIENT-NORM ANALYSE")
    print("=" * 80)

    grad_tags = [t for t in scalar_tags if 'grad_norm' in t]

    if not grad_tags:
        print("\n   ❌ PROBLEM: Keine Gradient-Norm Metriken gefunden!")
        print("   → Stelle sicher dass 'tb_log_grad_norm: true' in der Config ist")
        return

    print(f"\n   Gefundene Gradient-Metriken:")
    for tag in sorted(grad_tags):
        print(f"     - {tag}")

    # Lade und analysiere die Werte
    print("\n" + "=" * 80)
    print("3. GRADIENT-WERTE (letzte 10 Steps)")
    print("=" * 80)

    for tag in sorted(grad_tags):
        events = ea.Scalars(tag)
        if not events:
            continue

        # Letzte 10 Werte
        recent = events[-10:]

        print(f"\n   {tag}:")
        print(f"   Total steps: {len(events)}")

        if len(events) > 0:
            values = [e.value for e in recent]
            steps = [e.step for e in recent]

            print(f"   Letzte Werte:")
            for step, val in zip(steps, values):
                print(f"     Step {step:5d}: {val:.6f}")

            # Statistiken
            all_values = [e.value for e in events]
            avg = sum(all_values) / len(all_values)
            min_val = min(all_values)
            max_val = max(all_values)

            print(f"\n   Statistiken (alle {len(events)} steps):")
            print(f"     Average: {avg:.6f}")
            print(f"     Min:     {min_val:.6f}")
            print(f"     Max:     {max_val:.6f}")

            # Warnungen
            if 'te1' in tag.lower() or 'text_encoder_1' in tag.lower():
                if avg < 0.1:
                    print(f"     ⚠ WARNING: TE1 Gradienten sind SEHR klein (avg={avg:.6f})")
                    print(f"     → Eventuell Learning Rate erhöhen")

            if 'te2' in tag.lower() or 'text_encoder_2' in tag.lower():
                if avg < 0.5:
                    print(f"     ⚠ WARNING: TE2 Gradienten sind klein (avg={avg:.6f})")

    # Learning Rates analysieren
    print("\n" + "=" * 80)
    print("4. LEARNING RATE ANALYSE")
    print("=" * 80)

    lr_tags = [t for t in scalar_tags if 'lr' in t.lower() and 'train/' in t]

    for tag in sorted(lr_tags):
        events = ea.Scalars(tag)
        if not events:
            continue

        recent = events[-5:]
        print(f"\n   {tag}:")
        print(f"   Aktuelle LR:")
        for e in recent:
            print(f"     Step {e.step:5d}: {e.value:.3e}")

    # Loss analysieren
    print("\n" + "=" * 80)
    print("5. LOSS ANALYSE")
    print("=" * 80)

    if 'train/loss' in scalar_tags:
        events = ea.Scalars('train/loss')

        # Erste und letzte Werte
        first_10 = events[:10]
        last_10 = events[-10:]

        print(f"\n   Train Loss (total steps: {len(events)}):")

        print(f"\n   Erste 10 Steps:")
        for e in first_10:
            print(f"     Step {e.step:5d}: {e.value:.6f}")

        print(f"\n   Letzte 10 Steps:")
        for e in last_10:
            print(f"     Step {e.step:5d}: {e.value:.6f}")

        # Check ob Loss sinkt
        if len(events) >= 100:
            first_100_avg = sum(e.value for e in events[:100]) / 100
            last_100_avg = sum(e.value for e in events[-100:]) / 100

            print(f"\n   Loss Trend:")
            print(f"     First 100 steps avg: {first_100_avg:.6f}")
            print(f"     Last 100 steps avg:  {last_100_avg:.6f}")
            print(f"     Improvement:         {((first_100_avg - last_100_avg) / first_100_avg * 100):.2f}%")

            if last_100_avg >= first_100_avg:
                print(f"     ⚠ WARNING: Loss sinkt NICHT! Eventuell Problem mit Training")

    print("\n" + "=" * 80)
    print("ANALYSE ABGESCHLOSSEN")
    print("=" * 80)

    # Zusammenfassung
    print("\n💡 ZUSAMMENFASSUNG:")

    has_te_grads = any('grad_norm_te' in t for t in scalar_tags)

    if has_te_grads:
        print("   ✓ Text Encoder Gradienten werden geloggt")

        # Check actual values
        for tag in ['train/grad_norm_te1', 'train/grad_norm_te2']:
            if tag in scalar_tags:
                events = ea.Scalars(tag)
                if events:
                    avg = sum(e.value for e in events) / len(events)
                    if avg < 0.1:
                        print(f"   ⚠ {tag} ist sehr klein ({avg:.6f}) - eventuell LR erhöhen")
    else:
        print("   ❌ KEINE Text Encoder Gradienten gefunden")
        print("   → Setze 'tensorboard.log_grad_norm: true' in der Config")

if __name__ == "__main__":
    import sys
    run_name = sys.argv[1] if len(sys.argv) > 1 else "Chris_v12_te_fixed"
    analyze_tensorboard_logs(run_name)
