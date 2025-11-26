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
- Für TensorBoard-Monitoring: `pip install tensorboard` (im aktuellen PyTorch-Stack meist bereits enthalten)

## Konfiguration

Alle wichtigen Parameter liegen in `config.json`. Wichtige Gruppen:

### `model`
- `id`: Pfad oder HuggingFace-ID des Basis-SDXL-Modells
- `use_ema`, `ema_decay`, `use_bf16`, `use_gradient_checkpointing`: Trainings-Optimierungen
- `train_text_encoder_1` / `train_text_encoder_2`: steuern, ob der jeweilige CLIP-Encoder trainiert wird (Defaults: TE1 `true`, TE2 `false`); setze sie explizit auf `false`, um einen oder beide Encoder einzufrieren.
- `use_torch_compile`: aktiviert `torch.compile` (Inductor). Im Zweifel erst testen, manche Kombinationen funktionieren nicht stabil.
- `torch_compile_kwargs`: optionale Dictionary-Parameter für `torch.compile` (z. B. `{ "mode": "max-autotune" }`)
- `ema_decay`: Glättungsfaktor für die EMA-Gewichte (zwischen 0 und 1, Standard 0.9999). Wird nur genutzt, wenn `use_ema=true`.

### `run`
- `name`: Klarer Name für den aktuellen Lauf (z. B. `new_beginning_v_0_0_4`)
- `output_root`: Basisordner für Diffusers-Outputs (Standard `.output`)
- `checkpoint_root`: Basisordner für Single-File-Checkpoints (Standard `<output_root>/safetensors`)

> Wenn `training.output_dir` bzw. `export.checkpoint_path` leer bleiben, werden sie automatisch aus `run.*` abgeleitet (`{output_root}/{name}` und `{checkpoint_root}/{name}.safetensors`).

### `training`
- `output_dir`: Zielordner für den Diffusers-Checkpoint. Kann leer bleiben, wenn `run.name` gesetzt ist.
- `num_steps` oder `num_epochs`: Stoppbedingung (mindestens eine Angabe erforderlich)
- `lr_unet`, `lr_text_encoder_1`, `lr_text_encoder_2`: Lernraten für UNet sowie beide Text-Encoder (TE2 optional). Die Encoder-Werte müssen explizit gesetzt werden, wenn sie trainiert werden sollen.
- `seed`: Fixiert alle RNGs (Python, NumPy, Torch CPU/GPU). `null` oder `0` -> zufälliger Seed pro Run.
- `grad_accum_steps`, `batch_size`: Steuerung des effektiven Batchsizes
- `noise_offset`: Stärke des Noise-Offsets (z. B. 0.1)
- `min_sigma` & `min_sigma_warmup_steps`: Begrenzen sehr kleine Sigmas (z. B. 0.4 mit 500 Warmup-Schritten); Einsätze greifen erst nach dem Warmup und ändern niemals die Scheduler-Timesteps, sondern nur die Loss-Gewichtung.
- `prediction_type`: z. B. `"v_prediction"` zum V-Pred-Fix, ansonsten `null` für Scheduler-Default
- `snr_gamma`: aktiviert das Min-SNR-gewichtete Loss (z. B. 5.0–7.0) für stabilere Gradienten.
- `max_grad_norm`: Gradient-Clipping pro Optimizer-Step (z. B. 1.0). `null` deaktiviert das Feature.
- `detect_anomaly`: bricht das Training ab, sobald NaN/Inf im Loss auftauchen.
- `lr_warmup_steps`: Lineares Warmup der Lernrate über die ersten N Optimizer-Schritte (typisch 100–500).
- `lr_scheduler`: konfigurierbarer LR-Faktor (`type`: `constant`, `linear_decay`, `cosine_decay`, `warmup_steps`, `min_factor`). Der Faktor wird pro globalem Step berechnet und auf alle Param-Gruppen angewendet (Basis-LR * Faktor). Wird dieser Block nicht gesetzt, bleibt das Verhalten wie bisher (keine Decays; Legacy-Warmup über `lr_warmup_steps`).
- `ema_update_every`: Wie oft (in Steps) die EMA nach `lr_warmup_steps` aktualisiert wird; Standard 10. Davor bleibt EMA inaktiv.
- `resume_from`: Pfad zu einem bestehenden Diffusers-Output, dessen Gewichte weitertrainiert werden sollen.
- `resume_state_path`: Optionaler Pfad zu einer gespeicherten Trainer-State-Datei (`trainer_state.pt`). Wenn nicht gesetzt, wird bei `resume_from` automatisch `<resume_from>/trainer_state.pt` verwendet.
- `state_path`: Zielpfad für den Trainer-State des aktuellen Laufs (Standard: `<output_dir>/trainer_state.pt`). Enthält Optimizer-, Scaler-, EMA- und LR-Scheduler-Status.
- `tensorboard`: Block zur Aktivierung des TensorBoard-Loggings, z. B.:
  ```json
  "tensorboard": {
    "enabled": true,
    "base_dir": "./logs/tensorboard",
    "log_grad_norm": true,
    "log_scaler": true
  }
  ```
  - `log_dir`: optionaler fester Pfad; wenn leer, wird `{base_dir}/{run.name}` verwendet.
  - `log_grad_norm`: loggt die grad-Norm (erfordert `scaler.unscale_`, daher minimaler Overhead).
  - `log_scaler`: schreibt den aktuellen AMP-Scale-Wert.

### `data`
- `image_dir`: Ordner mit Bildern (PNG/JPG/WebP)
- `caption_dropout_prob`: Wahrscheinlichkeit, Captions komplett zu droppen (0–1)
- `caption_shuffle_prob`: Wahrscheinlichkeit, Teil-Captions neu zu mischen
- `caption_shuffle_separator`: Trennzeichen zur Tokenisierung (Standard `","`)
- `caption_shuffle_min_tokens`: Mindestanzahl Tokens, bevor Shuffle greift
- Der Dataset-Loader erhält jedes Bild in Originalauflösung, wählt anhand des Seitenverhältnisses den besten Bucket und skaliert erst dann auf die Zielauflösung; ohne Buckets wird nur die längste Seite auf `size` begrenzt. Ausgelieferte Tensors liegen in `float16`, um RAM zu sparen.
- `bucket`: Bucketed Training aktivieren (z. B. mehrere Auflösungen):
  ```json
  "bucket": {
    "enabled": true,
    "resolutions": [[1024,1024],[896,1152],[1152,896]],
    "divisible_by": 64,
    "batch_size": 2,
    "drop_last": true,
    "per_resolution_batch_sizes": {
      "832x1216": 2,
      "1216x832": 2
    }
  }
  ```
  Jede Auflösung bildet einen eigenen Bucket; der Dataloader gruppiert automatisch nur Bilder gleicher Auflösung in einen Batch, nutzt optional eine pro-Bucket-Batchsize und loggt beim Start die komplette Verteilung inkl. effektiver Batchgrößen.
- `latent_cache`: speichert vorab berechnete VAE-Latents auf Disk. Beispiel:
  ```json
  "latent_cache": {
    "enabled": true,
    "cache_dir": "./cache/latents",
    "dtype": "auto",
    "build_batch_size": 2
  }
  ```
  Beim Start prüft das Skript, ob alle Latents vorhanden sind und erzeugt fehlende (per VAE) bevor das Training losläuft. Latents werden in der jeweiligen Bucket-Auflösung (`H/8 × W/8`) abgelegt; während des Trainings werden dann nur noch die `.safetensors`-Latents geladen – keine erneute VAE-Passage notwendig.

### `optimizer`
- AdamW (8-Bit via bitsandbytes); Parameter z. B. `weight_decay`, `betas`, `eps`

### `export`
- `save_single_file`: Schaltet die Erstellung einer `.safetensors`-Checkpointdatei ein/aus
- `checkpoint_path`: Zielpfad. Kann leer bleiben, wenn `run.name` gesetzt ist.
- `converter_script`: lokaler Pfad zum SDXL-Kompatibilitäts-Skript (`./converttosdxl.py`). Falls du ein anderes Skript einsetzen willst, hier dessen Pfad hinterlegen (Pflichtfeld).
- `half_precision`, `use_safetensors`, `extra_args`: Feineinstellungen für den Konverter

### `eval`
- Gemeinsame Felder: `backend` (`diffusers` oder `kdiffusion`), `sampler_name` (z. B. `dpmpp_2m_sde_heun`), `scheduler` (Diffusers-Schedulername oder Shortcuts wie `beta`, `euler`, `lms`), `num_inference_steps`, `cfg_scale`, `prompts_path` (JSON mit `[{"prompt": ..., "negative_prompt": ..., "seed": ..., "height": ..., "width": ...}]`, Beispiel in `data/eval_prompts.example.json`), optionale Default-Auflösung `height`/`width` und `use_ema`, um während der Eval temporär EMA-Gewichte auf den UNet zu kopieren.
- Pro Prompt können `height`/`width` oder ein `resolution`/`size`-String wie `1024x640` gesetzt werden; sie überschreiben die globalen Defaults.
- `live`: `enabled`, `every_n_steps` und optional `max_batches`, um pro Zwischen-Eval nur die ersten N Prompts zu rendern. Ergebnisse landen unter `.output/<run>/eval/live/step_<global_step>/`.
- `final`: aktiviert einen Abschluss-Eval nach dem letzten Trainingstep; optional eigenes `max_batches`. Ausgabe unter `.output/<run>/eval/final/`.
- Während `run_eval()` werden alle relevanten Module in `eval()` versetzt, `torch.inference_mode()` aktiviert und nach Abschluss der ursprüngliche Train-Zustand wiederhergestellt. TensorBoard erhält einen kurzen Text-Eintrag (`eval/live` bzw. `eval/final`).
- Für `backend: kdiffusion` nutzt der Trainer intern die `StableDiffusionXLKDiffusionPipeline` aus diffusers und propagiert Sampler/Scheduler-Einstellungen 1:1 weiter (siehe zusätzliche Abhängigkeit `k-diffusion` in `requirements.txt`).

### Stabilität & Monitoring
- Min-SNR-gewichtetet Loss (`training.snr_gamma`) reduziert den Einfluss von extrem verrauschten Schritten.
- `training.max_grad_norm` aktiviert Gradient-Clipping (z. B. 1.0), `training.detect_anomaly` bricht bei NaN/Inf sofort ab.
- `training.lr_warmup_steps` erlaubt lineares Warmup der Lernrate (empfohlen 100–500 Schritte).
- `training.lr_scheduler`: neuer Block für skalare LR-Faktoren (`type`: `constant`, `linear_decay`, `cosine_decay`, `cosine_restarts`, `warmup_steps`, `min_factor`). Der Faktor wird auf alle Param-Gruppen (UNet/TEs) angewendet; während des Warmups geht er linear von 0 → 1, anschließend entweder konstant oder mit dem gewählten Decay weiter. Wenn der Block fehlt, verhält sich der Trainer wie früher (keine Decays; optionales Legacy-Warmup über `training.lr_warmup_steps`).
  - `cosine_restarts` führt zyklische Cosine-Decay-Phasen aus (SGDR-Style): `cycle_steps` definiert die Länge eines Zyklus nach dem Warmup, `cycle_mult` (>1.0 → Zyklen werden länger, =1.0 → konstante Länge). Beispiel:
    ```json
    "lr_scheduler": {
      "type": "cosine_restarts",
      "warmup_steps": 200,
      "min_factor": 0.3,
      "cycle_steps": 400,
      "cycle_mult": 1.0
    }
    ```
- Beim Start werden – sofern Buckets aktiv sind – die tatsächlichen Bucket-Verteilungen inklusive effektiver Batchgrößen geloggt.
- Für TensorBoard: `tensorboard --logdir ./logs/tensorboard` starten (oder den in der Config gesetzten Pfad) und im Browser öffnen. Es werden Loss, Lernraten (UNet/Text-Encoder), Grad-Norm (optional), AMP-Scale, Bucket-Verteilung sowie Basis-Metadaten (Effektiv-Batchsize, Anzahl Samples) geloggt.

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
- Nach Abschluss (und optionaler EMA-Übernahme) wird das Modell gespeichert, der Trainer-State (`trainer_state.pt`) aktualisiert und – falls aktiviert – der single-file Konverter gestartet. Für Resume einfach `training.resume_from` und ggf. `resume_state_path` setzen.

## Ergebnis nutzen

- Diffusers-Ordner (`output_dir`) kann direkt mit `StableDiffusionXLPipeline.from_pretrained(...)` geladen werden.
- Die erzeugte `.safetensors`-Datei ist kompatibel mit ComfyUI (Ordner `ComfyUI/models/checkpoints/`).

## Tipps

- Für kleine Tests `num_steps` deutlich reduzieren, `log_every`/`checkpoint_every` entsprechend setzen.
- Caption-Augmentation lässt sich schnell testen, indem man z. B. `caption_dropout_prob` auf 0.1 und `caption_shuffle_prob` auf 0.2 stellt.
- Noise-Offset (~0.1) + MinSigma (~0.4 mit ausreichendem Warmup) helfen oft bei kontrastreichen Motiven, ohne die Noise-Schedule zu zerstören.
- Bei Trainingsabbrüchen durch toten FlashAttention-Fallback ggf. `flash_attn` entfernen oder PyTorch SDPA nutzen (passiert automatisch). Für GPUs ohne H100/Blackwell-Unterstützung empfiehlt sich FlashAttention 2.x oder der Standard-SDP-Pfad.

Viel Erfolg beim Fine-Tuning! Bei Fragen oder Wunsch nach zusätzlichen Features einfach melden.

## Demo: Trainings-Forward-Pass visualisieren

- `demo_training_flow.py` lädt dieselben Komponenten wie im Training (UNet/VAE/TE1+2, Scheduler, VAE-Scaling) anhand deiner `config.json`, zieht 3–4 Beispielsamples aus `data/images` (inkl. Caption-Transforms) und simuliert genau den Forward-Schritt ohne Backprop.
- Starten mit aktivierter venv: `python demo_training_flow.py` (Optionen: `--config pfad/zur/config.json`, `--samples 3`), UI öffnet sich via Gradio.
- UI: Sample-Dropdown + Timestep-Slider → zeigt nebeneinander das trainierte Input-Bild (nach Resize/Norm), `x_t` mit Noise + Scheduler sowie `x0_pred`, also die UNet-Schätzung des sauberen Bildes. Zusätzlich gibt es eine Cross-Attention-Heatmap pro auswählbarem Token (Aggregat über die hochauflösende Cross-Attn). Läuft unter `torch.no_grad()`, nutzt GPU falls verfügbar und fällt ansonsten auf CPU zurück.

