# SDXL Fine-Tuner (Flash & Feature Friendly)

Dieses Projekt enthält ein leichtgewichtiges Trainings-Setup, um eigene Stable-Diffusion-XL-Modelle unter WSL2 in einer virtuellen Umgebung zu fine-tunen. Der Fokus liegt auf schnellen Experimenten mit FlashAttention, 8-Bit-Optimierung und praktischen Zusatzfeatures wie Caption-Dropout/-Shuffle, Noise-Offset oder V-Prediction-Fix.

## Inhalt

- `train_sdxl_ff.py` – Hauptskript, das eine SDXL-Pipeline lädt, Daten vorbereitet, trainiert, optional EMA nutzt und nach jedem Lauf sowohl Diffusers-Ordner als auch (falls konfiguriert) eine `.safetensors`-Checkpoint-Datei erzeugt.
- `dataset.py` – Minimaler Dataset-Wrapper, der Bilder + `.txt`-Captions liest und Caption-Augmentation (Dropout & Shuffle) unterstützt.
- `config.json` – zentrale Konfigurationsdatei für Device, Modell-ID, Trainingsparameter, Datenpfad, Optimizer, Export-Einstellungen und die erwähnten Zusatzfeatures.

## Voraussetzungen

- WSL2 mit Ubuntu
- Python 3.12 (bereits in `venv/` vorbereitet)
- CUDA-fähige GPU (für FlashAttention/AMP erforderlich)
- Abhängigkeiten sind im virtuellen Environment installiert (`pip install -r requirements.txt` falls vorhanden)

## Konfiguration

Alle wichtigen Parameter liegen in `config.json`. Wichtige Gruppen:

### `model`
- `id`: Pfad oder HuggingFace-ID des Basis-SDXL-Modells
- `use_ema`, `use_bf16`, `use_gradient_checkpointing`: Trainings-Optimierungen
- `train_text_encoders`: `true/false`, ob CLIP/Text-Encoder weitertrainiert oder eingefroren werden sollen
- `use_torch_compile`: aktiviert `torch.compile` (Inductor). Im Zweifel erst testen, manche Kombinationen funktionieren nicht stabil.
- `torch_compile_kwargs`: optionale Dictionary-Parameter für `torch.compile` (z. B. `{ "mode": "max-autotune" }`)

### `run`
- `name`: Klarer Name für den aktuellen Lauf (z. B. `new_beginning_v_0_0_4`)
- `output_root`: Basisordner für Diffusers-Outputs (Standard `.output`)
- `checkpoint_root`: Basisordner für Single-File-Checkpoints (Standard `<output_root>/safetensors`)

> Wenn `training.output_dir` bzw. `export.checkpoint_path` leer bleiben, werden sie automatisch aus `run.*` abgeleitet (`{output_root}/{name}` und `{checkpoint_root}/{name}.safetensors`).

### `training`
- `output_dir`: Zielordner für den Diffusers-Checkpoint. Kann leer bleiben, wenn `run.name` gesetzt ist.
- `num_steps` oder `num_epochs`: Stoppbedingung (mindestens eine Angabe erforderlich)
- `lr_unet`, `lr_text_encoder`: Lernraten für UNet und beide Text-Encoder
- `grad_accum_steps`, `batch_size`: Steuerung des effektiven Batchsizes
- `noise_offset`: Stärke des Noise-Offsets (z. B. 0.1)
- `min_sigma` & `min_sigma_warmup_steps`: Begrenzen sehr kleine Sigmas (z. B. 0.4 mit 500 Warmup-Schritten)
- `prediction_type`: z. B. `"v_prediction"` zum V-Pred-Fix, ansonsten `null` für Scheduler-Default

### `data`
- `image_dir`: Ordner mit Bildern (PNG/JPG/WebP)
- `caption_dropout_prob`: Wahrscheinlichkeit, Captions komplett zu droppen (0–1)
- `caption_shuffle_prob`: Wahrscheinlichkeit, Teil-Captions neu zu mischen
- `caption_shuffle_separator`: Trennzeichen zur Tokenisierung (Standard `","`)
- `caption_shuffle_min_tokens`: Mindestanzahl Tokens, bevor Shuffle greift

### `optimizer`
- AdamW (8-Bit via bitsandbytes); Parameter z. B. `weight_decay`, `betas`, `eps`

### `export`
- `save_single_file`: Schaltet die Erstellung einer `.safetensors`-Checkpointdatei ein/aus
- `checkpoint_path`: Zielpfad. Kann leer bleiben, wenn `run.name` gesetzt ist.
- `converter_script`: lokaler Pfad zum SDXL-Kompatibilitäts-Skript (`./converttosdxl.py`). Falls du ein anderes Skript einsetzen willst, hier dessen Pfad hinterlegen (Pflichtfeld).
- `half_precision`, `use_safetensors`, `extra_args`: Feineinstellungen für den Konverter

## Training starten

1. Virtuelle Umgebung aktivieren:
   ```bash
   source venv/bin/activate
   ```
2. Config anpassen (`config.json`).
3. Training ausführen:
   ```bash
   python train_sdxl_ff.py
   ```

Während des Trainings:
- Fortschritt via `tqdm`-Statusleiste
- Optionales Logging/Checkpointing gemäß Config
- Nach Abschluss (und optionaler EMA-Übernahme) wird das Modell gespeichert und – falls aktiviert – der single-file Konverter gestartet.

## Ergebnis nutzen

- Diffusers-Ordner (`output_dir`) kann direkt mit `StableDiffusionXLPipeline.from_pretrained(...)` geladen werden.
- Die erzeugte `.safetensors`-Datei ist kompatibel mit ComfyUI (Ordner `ComfyUI/models/checkpoints/`).

## Tipps

- Für kleine Tests `num_steps` deutlich reduzieren, `log_every`/`checkpoint_every` entsprechend setzen.
- Caption-Augmentation lässt sich schnell testen, indem man z. B. `caption_dropout_prob` auf 0.1 und `caption_shuffle_prob` auf 0.2 stellt.
- Noise-Offset (~0.1) + MinSigma (~0.4) helfen oft bei kontrastreichen Motiven.
- Bei Trainingsabbrüchen durch toten FlashAttention-Fallback ggf. `flash_attn` entfernen oder PyTorch SDPA nutzen (passiert automatisch).

Viel Erfolg beim Fine-Tuning! Bei Fragen oder Wunsch nach zusätzlichen Features einfach melden.

