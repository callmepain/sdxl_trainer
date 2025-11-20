import contextlib
import json
import random
import subprocess
import sys
import warnings
from pathlib import Path

import torch
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DEISMultistepScheduler,
    EDMEulerScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    UniPCMultistepScheduler,
)
from torch.utils.tensorboard import SummaryWriter


class EvalRunner:
    DIFFUSERS_SCHEDULER_ALIASES = {
        "beta": "EulerAncestralDiscreteScheduler",
        "euler": "EulerDiscreteScheduler",
        "euler_a": "EulerAncestralDiscreteScheduler",
        "heun": "HeunDiscreteScheduler",
        "lms": "LMSDiscreteScheduler",
        "kdpm2": "KDPM2DiscreteScheduler",
        "kdpm2_a": "KDPM2AncestralDiscreteScheduler",
        "dpmpp": "DPMSolverMultistepScheduler",
        "dpmsolver": "DPMSolverMultistepScheduler",
        "dpmsde": "DPMSolverSDEScheduler",
        "deis": "DEISMultistepScheduler",
        "uni_pc": "UniPCMultistepScheduler",
        "edm": "EDMEulerScheduler",
    }

    def __init__(
        self,
        pipeline,
        eval_cfg: dict,
        output_dir: Path,
        device,
        dtype,
        ema_unet=None,
        tb_writer: SummaryWriter | None = None,
        run_name: str | None = None,
        expected_final_step: int | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.eval_cfg = eval_cfg or {}
        self.live_cfg = self.eval_cfg.get("live") or {}
        self.final_cfg = self.eval_cfg.get("final") or {}
        self.backend = (self.eval_cfg.get("backend") or "diffusers").strip().lower()
        self.sampler_name = self.eval_cfg.get("sampler_name")
        self.scheduler_name = self.eval_cfg.get("scheduler")
        self.num_steps = int(self.eval_cfg.get("num_inference_steps") or 30)
        self.cfg_scale = float(self.eval_cfg.get("cfg_scale") or 7.5)
        self.prompts_path = self.eval_cfg.get("prompts_path")
        self.prompts = self._load_prompts(self.prompts_path)
        self.use_ema = bool(self.eval_cfg.get("use_ema", True))
        self.default_height = self._coerce_resolution_value(self.eval_cfg.get("height"))
        self.default_width = self._coerce_resolution_value(self.eval_cfg.get("width"))
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        self.dtype = dtype
        self.ema_unet = ema_unet
        self.tb_writer = tb_writer
        self.run_name = run_name or "run"
        self._kdiff_pipe = None
        self._scheduler_cache = None
        self._generator_device = self.device if self.device.type != "mps" else torch.device("cpu")
        self.expected_final_step = expected_final_step
        self._final_ran = False

        self.live_every = None
        self.live_enabled = False
        every = self.live_cfg.get("every_n_steps")
        if self.live_cfg.get("enabled") and every:
            try:
                self.live_every = max(1, int(every))
                self.live_enabled = True
            except (TypeError, ValueError):
                warnings.warn("Eval live.every_n_steps ist ungültig – Live-Eval deaktiviert.", stacklevel=2)
                self.live_enabled = False

        self.final_enabled = bool(self.final_cfg.get("enabled"))

    def _load_prompts(self, path_value):
        if not path_value:
            return []
        path = Path(path_value).expanduser()
        if not path.exists():
            warnings.warn(f"Eval-Promptdatei {path} wurde nicht gefunden.", stacklevel=2)
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            warnings.warn(f"Eval-Prompts konnten nicht gelesen werden ({err}).", stacklevel=2)
            return []
        if isinstance(data, dict) and "prompts" in data:
            data = data["prompts"]
        prompts = []
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, str):
                    prompts.append({"prompt": entry, "negative_prompt": None, "seed": None})
                elif isinstance(entry, dict):
                    prompt_text = entry.get("prompt")
                    if not prompt_text:
                        continue
                    prompts.append(
                        {
                            "prompt": prompt_text,
                            "negative_prompt": entry.get("negative_prompt") or entry.get("negative"),
                            "seed": entry.get("seed"),
                        }
                    )
        return prompts

    def has_work(self) -> bool:
        return bool(self.prompts) and (self.live_enabled or self.final_enabled)

    def maybe_run_live(self, global_step: int, final_pending: bool = False) -> None:
        if not (self.live_enabled and self.prompts):
            return
        if global_step <= 0 or self.live_every is None:
            return
        if final_pending:
            return
        if self.expected_final_step is not None and global_step >= self.expected_final_step:
            return
        if self._final_ran:
            return
        if global_step % self.live_every != 0:
            return
        max_batches = self._coerce_int(self.live_cfg.get("max_batches"))
        self._run_eval(eval_type="live", step=global_step, max_batches=max_batches)

    def run_final(self, global_step: int) -> None:
        if not (self.final_enabled and self.prompts):
            return
        max_batches = self._coerce_int(self.final_cfg.get("max_batches"))
        self._final_ran = True
        self._run_eval(eval_type="final", step=global_step, max_batches=max_batches)

    def _coerce_int(self, value):
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        return parsed

    def _coerce_resolution_value(self, value):
        if value is None:
            return None
        token = value
        if isinstance(token, str):
            token = token.strip().lower().rstrip("px")
            if "x" in token or token == "":
                return None
        try:
            parsed = int(float(token))
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        aligned = int(round(parsed / 8.0) * 8)
        return max(64, aligned)

    def _parse_resolution_string(self, value):
        if not value or not isinstance(value, str):
            return None
        token = value.lower().replace("px", "").replace(" ", "")
        if "x" not in token:
            return None
        parts = token.split("x", 1)
        if len(parts) != 2:
            return None
        width = self._coerce_resolution_value(parts[0])
        height = self._coerce_resolution_value(parts[1])
        if width is None or height is None:
            return None
        return width, height

    def _resolve_prompt_resolution(self, entry):
        width = self._coerce_resolution_value(entry.get("width"))
        height = self._coerce_resolution_value(entry.get("height"))
        if (width is None or height is None) and isinstance(entry.get("resolution"), str):
            parsed = self._parse_resolution_string(entry.get("resolution"))
            if parsed is not None:
                width = width or parsed[0]
                height = height or parsed[1]
        if (width is None or height is None) and isinstance(entry.get("size"), str):
            parsed = self._parse_resolution_string(entry.get("size"))
            if parsed is not None:
                width = width or parsed[0]
                height = height or parsed[1]
        if width is None:
            width = self.default_width
        if height is None:
            height = self.default_height
        if width is None or height is None:
            return None, None
        return height, width

    def _run_eval(self, eval_type: str, step: int, max_batches: int | None) -> None:
        selected = self.prompts
        if max_batches is not None:
            selected = selected[: max_batches]
        if not selected:
            return
        dest = self._build_eval_dir(eval_type, step)
        dest.mkdir(parents=True, exist_ok=True)
        self._log_eval_start(eval_type, step, len(selected))
        with self._temporary_eval_context():
            with self._maybe_ema_weights():
                if self.backend == "kdiffusion":
                    self._generate_with_kdiffusion(selected, dest, eval_type, step)
                else:
                    self._generate_with_diffusers(selected, dest, eval_type, step)
        self._cleanup_after_eval()

    def _build_eval_dir(self, eval_type: str, step: int) -> Path:
        base = self.output_dir / "eval" / eval_type
        if eval_type == "live":
            return base / f"step_{step:06d}"
        return base

    def _log_eval_start(self, eval_type: str, step: int, batch_count: int) -> None:
        res_info = ""
        if self.default_width and self.default_height:
            res_info = f" {self.default_width}x{self.default_height}"
        msg = (
            f"[Eval:{eval_type}] step={step} backend={self.backend} sampler={self.sampler_name or '-'} "
            f"scheduler={self.scheduler_name or '-'}{res_info} batches={batch_count}"
        )
        print(msg)
        if self.tb_writer is not None:
            self.tb_writer.add_text(f"eval/{eval_type}", msg, step)

    @contextlib.contextmanager
    def _temporary_eval_context(self):
        modules = [
            self.pipeline.unet,
            getattr(self.pipeline, "text_encoder", None),
            getattr(self.pipeline, "text_encoder_2", None),
            getattr(self.pipeline, "vae", None),
        ]
        states = []
        for module in modules:
            if module is None:
                states.append(None)
                continue
            states.append(module.training)
            module.eval()
        try:
            with torch.inference_mode():
                yield
        finally:
            for module, state in zip(modules, states):
                if module is not None and state is not None:
                    module.train(state)

    @contextlib.contextmanager
    def _maybe_ema_weights(self):
        if not (self.use_ema and self.ema_unet is not None):
            yield
            return
        self.ema_unet.store(self.pipeline.unet.parameters())
        self.ema_unet.copy_to(self.pipeline.unet.parameters())
        try:
            yield
        finally:
            self.ema_unet.restore(self.pipeline.unet.parameters())

    def _generate_with_diffusers(self, prompts, dest: Path, eval_type: str, step: int) -> None:
        pipe = self.pipeline
        original_scheduler = pipe.scheduler
        scheduler_instance = self._create_scheduler_instance()
        if scheduler_instance is not None:
            pipe.scheduler = scheduler_instance
        pipe.set_progress_bar_config(disable=True)
        for idx, entry in enumerate(prompts):
            seed = entry.get("seed")
            generator = torch.Generator(device=self._generator_device)
            if seed is None:
                seed = random.randint(0, 2**31 - 1)
            generator.manual_seed(int(seed))
            negative_prompt = entry.get("negative_prompt")
            height, width = self._resolve_prompt_resolution(entry)
            call_kwargs = {
                "prompt": entry["prompt"],
                "negative_prompt": negative_prompt,
                "num_inference_steps": self.num_steps,
                "guidance_scale": self.cfg_scale,
                "generator": generator,
                "output_type": "pil",
            }
            if height is not None and width is not None:
                call_kwargs["height"] = height
                call_kwargs["width"] = width
            result = pipe(**call_kwargs)
            image = result.images[0]
            filename = dest / f"step_{step:06d}_idx_{idx:03d}_seed_{seed}.png"
            image.save(filename)
        if scheduler_instance is not None:
            pipe.scheduler = original_scheduler

    def _generate_with_kdiffusion(self, prompts, dest: Path, eval_type: str, step: int) -> None:
        kd_pipe = self._get_kdiffusion_pipeline()
        if kd_pipe is None:
            warnings.warn("k-diffusion Backend konnte nicht initialisiert werden. Fallback auf Diffusers.", stacklevel=2)
            self._generate_with_diffusers(prompts, dest, eval_type, step)
            return
        for idx, entry in enumerate(prompts):
            seed = entry.get("seed")
            generator = torch.Generator(device=self._generator_device)
            if seed is None:
                seed = random.randint(0, 2**31 - 1)
            generator.manual_seed(int(seed))
            negative_prompt = entry.get("negative_prompt")
            height, width = self._resolve_prompt_resolution(entry)
            call_kwargs = {
                "prompt": entry["prompt"],
                "negative_prompt": negative_prompt,
                "num_inference_steps": self.num_steps,
                "guidance_scale": self.cfg_scale,
                "generator": generator,
                "output_type": "pil",
            }
            if height is not None and width is not None:
                call_kwargs["height"] = height
                call_kwargs["width"] = width
            result = kd_pipe(**call_kwargs)
            image = result.images[0]
            filename = dest / f"step_{step:06d}_idx_{idx:03d}_seed_{seed}.png"
            image.save(filename)
        self._kdiff_pipe = None

    def _normalized_sampler_name(self):
        if not self.sampler_name:
            return None
        sampler = self.sampler_name.strip()
        if not sampler.startswith("sample_"):
            sampler = f"sample_{sampler}"
        return sampler

    def _get_kdiffusion_pipeline(self):
        if self._kdiff_pipe is not None:
            return self._kdiff_pipe
        try:
            from diffusers.pipelines.stable_diffusion_k_diffusion import (
                StableDiffusionXLKDiffusionPipeline,
            )
        except ImportError:
            warnings.warn(
                "StableDiffusionXLKDiffusionPipeline konnte nicht importiert werden. Ist k-diffusion installiert?",
                stacklevel=2,
            )
            return None
        scheduler_instance = self._create_scheduler_instance()
        if scheduler_instance is None:
            scheduler_instance = self.pipeline.scheduler.__class__.from_config(self.pipeline.scheduler.config)
        kd_pipe = StableDiffusionXLKDiffusionPipeline(
            vae=self.pipeline.vae,
            text_encoder=self.pipeline.text_encoder,
            text_encoder_2=self.pipeline.text_encoder_2,
            tokenizer=getattr(self.pipeline, "tokenizer", None),
            tokenizer_2=getattr(self.pipeline, "tokenizer_2", None),
            unet=self.pipeline.unet,
            scheduler=scheduler_instance,
        )
        sampler = self._normalized_sampler_name()
        if sampler:
            try:
                kd_pipe.set_scheduler(sampler)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"k-diffusion Sampler konnte nicht gesetzt werden ({exc}).", stacklevel=2)
        kd_pipe = kd_pipe.to(self.device)
        kd_pipe.set_progress_bar_config(disable=True)
        self._kdiff_pipe = kd_pipe
        return kd_pipe

    def _create_scheduler_instance(self):
        if not self.scheduler_name:
            return None
        scheduler_key = self.scheduler_name
        scheduler_cls_name = self.DIFFUSERS_SCHEDULER_ALIASES.get(scheduler_key.lower(), scheduler_key)
        scheduler_cls = globals().get(scheduler_cls_name)
        if scheduler_cls is None:
            warnings.warn(
                f"Unbekannter Scheduler {self.scheduler_name}. Verwende aktuellen Pipeline-Scheduler.",
                stacklevel=2,
            )
            return None
        try:
            return scheduler_cls.from_config(self.pipeline.scheduler.config)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Scheduler {scheduler_cls_name} konnte nicht initialisiert werden ({exc}).", stacklevel=2)
            return None

    def _cleanup_after_eval(self):
        if self.backend == "kdiffusion" and self._kdiff_pipe is not None:
            self._kdiff_pipe = None
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.empty_cache()


def _run_converter_script(script_path: Path, model_dir: Path, checkpoint_path: Path, cfg: dict) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_path),
        "--model_path",
        str(model_dir),
        "--checkpoint_path",
        str(checkpoint_path),
    ]

    if cfg.get("use_safetensors", True):
        cmd.append("--use_safetensors")
    if cfg.get("half_precision", True):
        cmd.append("--half")

    extra_args = cfg.get("extra_args") or []
    if not isinstance(extra_args, (list, tuple)):
        raise TypeError("`export.extra_args` muss eine Liste von zusätzlichen Argumenten sein.")
    cmd.extend(map(str, extra_args))

    print("Starte Diffusers-Konverter:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Der Diffusers-Konverter konnte nicht erfolgreich ausgeführt werden (Exit-Code {exc.returncode})."
        ) from exc


def export_single_file(model_dir: Path, cfg: dict) -> None:
    if not cfg.get("save_single_file", False):
        print("Single-File-Export deaktiviert.")
        return

    checkpoint_path = Path(cfg.get("checkpoint_path", f"{model_dir}.safetensors")).resolve()
    converter_path = cfg.get("converter_script")
    if not converter_path:
        raise ValueError("`export.converter_script` ist nicht gesetzt. Bitte Pfad zum Konverter angeben.")
    converter_path = Path(converter_path).expanduser().resolve()
    if not converter_path.exists():
        raise FileNotFoundError(f"Konverter-Skript {converter_path} existiert nicht.")

    _run_converter_script(converter_path, Path(model_dir).resolve(), checkpoint_path, cfg)
    print(f"Single-File-Checkpoint gespeichert unter: {checkpoint_path}")


__all__ = ["EvalRunner", "export_single_file"]
