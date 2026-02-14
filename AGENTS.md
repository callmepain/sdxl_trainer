# Repository Guidelines

## Project Structure & Module Organization
- `train_sdxl_ff.py` — main entry; loads config, prepares dataset/buckets, runs training, handles EMA, eval, export.
- `config*.json`, `config_utils.py` — presets and loader; keep personal edits in `config.json` (git-ignored). `config_decay.json` / `config.example.json` show reference setups.
- `dataset.py` — image+caption loader with dropout/shuffle and bucketed batching; expects images under `data/images` with optional `.txt` captions.
- `eval_export.py` — live/final evaluation; prompts in `data/eval_prompts*.json`; renders to `.output/<run>/eval/...`.
- `optim_utils.py`, `state_utils.py`, `converttosdxl.py`, `caption_cleanup.py`, `debug/` — optimizer helpers, checkpoint/state handling, SDXL converter, caption cleanup CLI, and ad-hoc probes. Large artifacts live in `.output/`, `cache/`, `logs/` (ignored).

## Build, Test, and Development Commands
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # install deps
cp config.example.json config.json       # personalize defaults
python train_sdxl_ff.py                  # run training with config.json
tensorboard --logdir ./logs/tensorboard  # monitor metrics
```
- Resume a run via `training.resume_from` / `resume_state_path` in `config.json`.
- Quick dataset sanity: `python caption_cleanup.py --help` or small eval prompts in `data/eval_prompts.example.json`.

## Coding Style & Naming Conventions
- Python 3.12; 4-space indentation; prefer `pathlib.Path` for file paths.
- Keep config keys snake_case to mirror `DEFAULT_CONFIG`; trailing commas where they aid diffs.
- Avoid prints in hot loops; use existing logging/warnings/tqdm patterns. Add type hints where present in helpers.

## Testing Guidelines
- No formal suite; do a smoke run with tiny settings (`training.num_steps=50`, reduced batch size) before long jobs.
- Enable live eval (`eval.live.enabled=true`, small prompt set) to validate sampler settings; check outputs in `.output/<run>/eval/live/`.
- Confirm TensorBoard logs populate under `logs/tensorboard` after changes touching logging.

## Commit & Pull Request Guidelines
- Commit messages: short, imperative, lower-case (e.g., `add lr scheduler`, `fix bucket logging`).
- PRs should note: brief summary, key config deltas, commands run (e.g., `python train_sdxl_ff.py`, `tensorboard`), and links/paths to outputs or eval renders.
- If altering training defaults or export behavior, mention backward compatibility and whether existing checkpoints remain usable.

## Security & Data Handling
- Do not commit model weights, generated images, or private datasets (`data/`, `.output/`, `.safetensors`, `.ckpt` are ignored). Keep tokens in env vars.
- Validate `run.name`/paths before long runs to avoid overwriting previous experiments; prefer unique names per run.
- Large artifacts stay untracked; clean via `rm -r .output/<run>` only when sure backups exist.

## System used for training
- AMD Threadripper 9960x 24 cores 48 threads
- 64 gb ddr5 ecc ram
- NVME ssd's pcie x4 gen 4
- 1x Geforce RTX 5090 32gb vram
- 1x Geforce GTX 5070ti 16gb vram (used for desktop)