# Repository Guidelines

## Project Structure & Module Organization
- `train_sdxl_ff.py`: main entry; loads config, prepares dataset/buckets, runs training, optional EMA, live/final eval, and export.
- `config*.json` and `config_utils.py`: default values and loader; keep personal settings in `config.json` (ignored). `config_decay.json` and `config.example.json` are reference presets.
- `dataset.py`: image + caption loader with dropout/shuffle and bucketed batching. Expects images in `data/images` with optional `.txt` captions.
- `eval_export.py`: evaluation runner; uses prompts from `data/eval_prompts*.json` and writes renders under `.output/<run>/eval/...`.
- `optim_utils.py`, `state_utils.py`, `converttosdxl.py`, `caption_cleanup.py`: optimizer helpers, checkpoint/state handling, SDXL converter, and caption cleanup CLI. `debug/` holds ad-hoc probes; `.output/`, `cache/`, `logs/`, and large artifacts stay untracked (see `.gitignore`).

## Setup, Build, and Development Commands
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# configure your run
cp config.example.json config.json  # edit as needed
python train_sdxl_ff.py             # starts training with config.json
tensorboard --logdir ./logs/tensorboard  # monitor metrics
```
`train_sdxl_ff.py` respects `run.*` and `training.*` paths, writes checkpoints to `.output/<run>/`, and can resume via `training.resume_from` / `resume_state_path`. `converttosdxl.py` converts a diffusers folder to `.safetensors` if invoked directly or via the export block.

## Coding Style & Naming Conventions
- Python 3.12, 4-space indentation, trailing commas where they aid diffs.
- Prefer `Path` over raw strings for file paths; keep config keys snake_case to match `DEFAULT_CONFIG`.
- Use type hints where present in helpers; keep functions small and data validation explicit (see `_coerce_*` helpers).
- Avoid printing inside hot loops; rely on `tqdm`, logging, or `warnings.warn` patterns used in the trainer.

## Testing Guidelines
- No formal test suite. Do a smoke run with small numbers (`training.num_steps=50`, lower batch size) before longer jobs.
- Enable live eval (`eval.live.enabled=true`) with a tiny prompt set to verify sampler settings; confirm outputs under `.output/<run>/eval/live/`.
- For dataset issues, run `caption_cleanup.py --help` or inspect `debug/` scripts; ensure TensorBoard logs populate when modifying logging code.

## Commit & Pull Request Guidelines
- Git history favors short, imperative, lower-case titles (e.g., `new structure`, `gitignore`); keep commits scoped.
- PRs should include: brief summary, key config deltas, commands run (`python train_sdxl_ff.py`, `tensorboard`), and links/paths to produced outputs or eval renders.
- If changing training defaults or export behavior, note backward-compatibility and whether existing checkpoints remain usable.

## Security & Data Handling
- Do not commit model weights, generated images, or private datasets (`data/`, `.output/`, `.safetensors`, `.ckpt` are ignored). Keep access tokens in env vars, not files.
- Validate paths before launching long runs to avoid overwriting prior experiments; prefer unique `run.name` values per experiment.
