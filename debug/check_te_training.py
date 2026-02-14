#!/usr/bin/env python3
"""
Diagnose-Script um zu überprüfen, ob die Text Encoder tatsächlich trainiert werden.
Führe dieses Script nach ein paar Trainingsschritten aus.
"""

import torch
from pathlib import Path
import json

def check_text_encoder_training():
    print("=" * 60)
    print("Text Encoder Training Diagnose")
    print("=" * 60)

    # 1. Config überprüfen
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)

        print("\n1. CONFIG CHECK:")
        print(f"   train_text_encoder_1: {cfg['model'].get('train_text_encoder_1')}")
        print(f"   train_text_encoder_2: {cfg['model'].get('train_text_encoder_2')}")
        print(f"   lr_text_encoder_1: {cfg['training'].get('lr_text_encoder_1')}")
        print(f"   lr_text_encoder_2: {cfg['training'].get('lr_text_encoder_2')}")

    # 2. Letzten Checkpoint überprüfen
    output_root = Path(".output")
    run_name = cfg.get("run", {}).get("name", "")

    if run_name:
        checkpoint_dirs = list(output_root.glob(f"{run_name}_step_*"))
        if not checkpoint_dirs:
            checkpoint_dirs = [output_root / run_name]

        if checkpoint_dirs:
            latest_checkpoint = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
            print(f"\n2. CHECKPOINT CHECK: {latest_checkpoint}")

            # Text Encoder 1 prüfen
            te1_path = latest_checkpoint / "text_encoder" / "model.safetensors"
            te2_path = latest_checkpoint / "text_encoder_2" / "model.safetensors"

            if te1_path.exists():
                print(f"   ✓ Text Encoder 1 vorhanden")
                import safetensors.torch
                te1_state = safetensors.torch.load_file(te1_path)
                print(f"     Parameter: {len(te1_state)} Tensoren")
                # Zeige ein paar Statistiken
                for key in list(te1_state.keys())[:3]:
                    tensor = te1_state[key]
                    print(f"     {key}: mean={tensor.float().mean().item():.6f}, std={tensor.float().std().item():.6f}")

            if te2_path.exists():
                print(f"   ✓ Text Encoder 2 vorhanden")
                import safetensors.torch
                te2_state = safetensors.torch.load_file(te2_path)
                print(f"     Parameter: {len(te2_state)} Tensoren")
                for key in list(te2_state.keys())[:3]:
                    tensor = te2_state[key]
                    print(f"     {key}: mean={tensor.float().mean().item():.6f}, std={tensor.float().std().item():.6f}")

    # 3. TensorBoard Logs überprüfen
    print("\n3. TENSORBOARD CHECK:")
    tb_base_dir = Path("./logs/tensorboard")
    if run_name:
        tb_run_dir = tb_base_dir / run_name
        if tb_run_dir.exists():
            print(f"   TensorBoard Log-Verzeichnis: {tb_run_dir}")
            print(f"   Führe aus: tensorboard --logdir {tb_run_dir}")
            print("   \n   Überprüfe folgende Metriken:")
            print("   - train/grad_norm_te1  (sollte > 0 sein wenn TE1 trainiert wird)")
            print("   - train/grad_norm_te2  (sollte > 0 sein wenn TE2 trainiert wird)")
            print("   - train/lr_text_encoder_1")
            print("   - train/lr_text_encoder_2")

            # Versuche TensorBoard Events zu lesen
            try:
                from tensorboard.backend.event_processing import event_accumulator

                event_files = list(tb_run_dir.glob("events.out.tfevents.*"))
                if event_files:
                    ea = event_accumulator.EventAccumulator(str(tb_run_dir))
                    ea.Reload()

                    print("\n   Verfügbare Metriken:")
                    for tag in sorted(ea.Tags()['scalars']):
                        if 'text_encoder' in tag or 'grad_norm_te' in tag:
                            events = ea.Scalars(tag)
                            if events:
                                last_val = events[-1].value
                                print(f"   - {tag}: {last_val:.6f} (letzter Wert)")
            except ImportError:
                print("   (tensorboard nicht installiert für detaillierte Analyse)")
            except Exception as e:
                print(f"   (Fehler beim Lesen der Events: {e})")

    # 4. Trainer State überprüfen
    print("\n4. OPTIMIZER STATE CHECK:")
    if run_name:
        state_path = output_root / run_name / "trainer_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            opt_state = state.get("optimizer", {})
            param_groups = opt_state.get("param_groups", [])

            print(f"   Optimizer Parameter-Gruppen: {len(param_groups)}")
            for i, pg in enumerate(param_groups):
                print(f"   Gruppe {i}: lr={pg.get('lr', 'n/a')}, params={len(pg.get('params', []))}")

            if len(param_groups) >= 3:
                print("\n   ✓ 3 Parameter-Gruppen gefunden (UNet + TE1 + TE2)")
            elif len(param_groups) == 2:
                print("\n   ⚠ Nur 2 Parameter-Gruppen (UNet + ein TE?)")
            elif len(param_groups) == 1:
                print("\n   ❌ Nur 1 Parameter-Gruppe (nur UNet!)")

    print("\n" + "=" * 60)
    print("DIAGNOSE ABGESCHLOSSEN")
    print("=" * 60)

    print("\n💡 EMPFEHLUNG:")
    print("Wenn du sicher sein willst, dass Training stattfindet:")
    print("1. Starte ein kurzes Training (50-100 steps)")
    print("2. Vergleiche die Checkpoint-Gewichte vorher/nachher")
    print("3. Oder überprüfe TensorBoard für grad_norm_te1/te2 > 0")

if __name__ == "__main__":
    check_text_encoder_training()
