# Repository Guidelines

## Project Structure & Module Organization
`train_sdxl_ff.py` is the entrypoint: it loads SDXL backbones, prepares optimizers/logging, and triggers exports. `dataset.py` contains caption-augmented datasets, bucket sampling, and latent caching logic—extend it instead of building custom loaders. `config.json` (and `config.example.json`) define every tunable option; point `data/` to your image + `.txt` caption pairs. Training outputs land in `.output/` and single-file checkpoints in `.output/safetensors/`, while `logs/` holds TensorBoard runs. `PainCraft_XL_Reforged_v1_diffusers/` provides a reference base model, `cache/` stores optional VAE latents, and `flash-attention/` mirrors vendor kernels. Use the existing `venv/` whenever editing or running scripts.

## Build, Test, and Development Commands
- `source venv/bin/activate` – enter the Python 3.12 environment with matching CUDA/BitsAndBytes.
- `python train_sdxl_ff.py` – run fine-tuning with values from `config.json`; update `run.name`, `data.image_dir`, and `training.*` before launch.
- `tensorboard --logdir ./logs/tensorboard` – monitor loss, grad norms, AMP scaler, and bucket stats per run.
- `python converttosdxl.py --model_path .output/<run> --checkpoint_path .output/safetensors/<run>.safetensors --half --use_safetensors` – export a ComfyUI/SDXL-compatible `.safetensors` file.

## Coding Style & Naming Conventions
Stick to idiomatic Python with 4-space indents, snake_case names, and `Path` usage for filesystem work. Follow the helper-heavy structure shown in `train_sdxl_ff.py`: private utilities prefixed with `_`, configuration dictionaries kept in lower_snake_case, and `f`-strings for logging. Type hints are encouraged on new public functions, and loggers or progress bars should leverage `tqdm` and TensorBoard writers already in place.

## Testing Guidelines
There is no pytest suite; rely on controlled runs. For code changes, execute a smoke job (`training.num_steps=50`, `batch_size=1`) to verify logging, checkpoint export, and EMA handling. When touching data pipelines, run with `data.bucket.enabled=true` to confirm bucket histograms appear and latent caches build without errors. Converter updates should be validated by re-running `converttosdxl.py` on a known `.output/<run>` folder and loading the result in ComfyUI or `StableDiffusionXLPipeline.from_single_file`.

## Commit & Pull Request Guidelines
History favors short, imperative subjects (`tensorboard`, `vae latent caching`). PRs must explain motivation, summarize config keys touched, and state the validation you ran (TensorBoard link, sample renders such as `test.png`). Reference related issues or experiments and mention GPU/driver requirements when they change so reviewers can reproduce quickly.

## Security & Configuration Tips
Do not commit personal datasets, HF tokens, or generated checkpoints. Scrub `config.json` paths and keep `.output/`, `cache/`, and `logs/` gitignored. When enabling FlashAttention or BF16, document the minimum driver/CUDA requirements in your PR description so other contributors can match the environment safely.
