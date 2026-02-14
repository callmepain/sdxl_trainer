#!/usr/bin/env python3
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from config_utils import DEFAULT_CONFIG, load_config


@dataclass(frozen=True)
class FieldSpec:
    tab: str
    section: str
    label: str
    path: tuple[str, ...]
    kind: str = "str"
    options: tuple[str, ...] = ()
    picker: str | None = None


@dataclass
class FieldBinding:
    var: tk.Variable
    spec: FieldSpec
    row: ttk.Frame


FIELD_SPECS = (
    FieldSpec("Run", "General", "Device", ("device",), "choice", ("cuda", "cpu")),
    FieldSpec("Run", "General", "Run Name", ("run", "name")),
    FieldSpec("Run", "Paths", "Output Root", ("run", "output_root"), "str", picker="dir"),
    FieldSpec("Run", "Paths", "Checkpoint Root", ("run", "checkpoint_root"), "str", picker="dir"),
    FieldSpec("Run", "Paths", "Model Path / ID", ("model", "id"), "str", picker="dir"),
    FieldSpec("Run", "Paths", "Image Directory", ("data", "image_dir"), "str", picker="dir"),
    FieldSpec("Run", "Paths", "Training Output Dir (optional)", ("training", "output_dir"), "str?", picker="dir"),
    FieldSpec("Run", "Paths", "Resume From (optional)", ("training", "resume_from"), "str?", picker="dir"),
    FieldSpec(
        "Run",
        "Paths",
        "Resume State Path (optional)",
        ("training", "resume_state_path"),
        "str?",
        picker="file",
    ),
    FieldSpec("Run", "Paths", "State Path (optional)", ("training", "state_path"), "str?", picker="save_file"),
    FieldSpec("Model", "Precision", "Use BF16", ("model", "use_bf16"), "bool"),
    FieldSpec("Model", "Precision", "Use EMA", ("model", "use_ema"), "bool"),
    FieldSpec("Model", "Precision", "EMA Decay", ("model", "ema_decay"), "float"),
    FieldSpec("Model", "Text Encoders", "Train Text Encoder 1", ("model", "train_text_encoder_1"), "bool"),
    FieldSpec("Model", "Text Encoders", "Train Text Encoder 2", ("model", "train_text_encoder_2"), "bool"),
    FieldSpec(
        "Model",
        "Performance",
        "Gradient Checkpointing",
        ("model", "use_gradient_checkpointing"),
        "bool",
    ),
    FieldSpec("Model", "Performance", "Torch Compile", ("model", "use_torch_compile"), "bool"),
    FieldSpec(
        "Model",
        "Attention",
        "Attention Backend",
        ("model", "attention_backend"),
        "choice",
        ("auto", "sdpa", "flash", "sage"),
    ),
    FieldSpec("Model", "Attention", "Sage Min Seq Length", ("model", "sage_min_seq_length"), "int"),
    FieldSpec("Training", "Steps", "Batch Size", ("training", "batch_size"), "int"),
    FieldSpec("Training", "Steps", "Grad Accum Steps", ("training", "grad_accum_steps"), "int"),
    FieldSpec("Training", "Steps", "Num Steps (optional)", ("training", "num_steps"), "int?"),
    FieldSpec("Training", "Steps", "Num Epochs (optional)", ("training", "num_epochs"), "int?"),
    FieldSpec("Training", "Rates", "UNet LR", ("training", "lr_unet"), "float"),
    FieldSpec("Training", "Rates", "TE1 LR (optional)", ("training", "lr_text_encoder_1"), "float?"),
    FieldSpec("Training", "Rates", "TE2 LR (optional)", ("training", "lr_text_encoder_2"), "float?"),
    FieldSpec("Training", "Rates", "Legacy LR Warmup Steps", ("training", "lr_warmup_steps"), "int"),
    FieldSpec("Training", "Noise", "Noise Offset", ("training", "noise_offset"), "float"),
    FieldSpec("Training", "Noise", "Min Sigma (optional)", ("training", "min_sigma"), "float?"),
    FieldSpec("Training", "Noise", "Min Sigma Warmup Steps", ("training", "min_sigma_warmup_steps"), "int"),
    FieldSpec("Training", "Noise", "Min Timestep (optional)", ("training", "min_timestep"), "int?"),
    FieldSpec("Training", "Noise", "Max Timestep (optional)", ("training", "max_timestep"), "int?"),
    FieldSpec(
        "Training",
        "Noise",
        "Prediction Type",
        ("training", "prediction_type"),
        "choice",
        ("none", "epsilon", "v_prediction", "sample"),
    ),
    FieldSpec("Training", "Stability", "SNR Gamma (optional)", ("training", "snr_gamma"), "float?"),
    FieldSpec("Training", "Stability", "Max Grad Norm (optional)", ("training", "max_grad_norm"), "float?"),
    FieldSpec("Training", "Stability", "Detect Anomaly", ("training", "detect_anomaly"), "bool"),
    FieldSpec("Training", "Stability", "EMA Update Every", ("training", "ema_update_every"), "int"),
    FieldSpec("Training", "Stability", "TE Freeze Fraction (optional)", ("training", "te_freeze_fraction"), "float?"),
    FieldSpec("Training", "Logging", "Checkpoint Every (optional)", ("training", "checkpoint_every"), "int?"),
    FieldSpec("Training", "Logging", "Log Every (optional)", ("training", "log_every"), "int?"),
    FieldSpec("Training", "Logging", "Seed (optional)", ("training", "seed"), "int?"),
    FieldSpec(
        "Scheduler",
        "Main",
        "Scheduler Type",
        ("training", "lr_scheduler", "type"),
        "choice",
        ("none", "constant", "cosine_decay", "linear_decay", "cosine_restarts"),
    ),
    FieldSpec("Scheduler", "Main", "Warmup Steps", ("training", "lr_scheduler", "warmup_steps"), "int"),
    FieldSpec("Scheduler", "Main", "Min Factor", ("training", "lr_scheduler", "min_factor"), "float"),
    FieldSpec("Scheduler", "Cosine Restarts", "Cycle Steps", ("training", "lr_scheduler", "cycle_steps"), "int"),
    FieldSpec("Scheduler", "Cosine Restarts", "Cycle Mult", ("training", "lr_scheduler", "cycle_mult"), "float"),
    FieldSpec("Optimizer", "AdamW", "Weight Decay", ("optimizer", "weight_decay"), "float"),
    FieldSpec("Optimizer", "AdamW", "Epsilon", ("optimizer", "eps"), "float"),
    FieldSpec("Optimizer", "AdamW", "Betas (JSON list)", ("optimizer", "betas"), "json"),
    FieldSpec("Data", "Core", "Image Size", ("data", "size"), "int"),
    FieldSpec("Data", "Core", "Shuffle", ("data", "shuffle"), "bool"),
    FieldSpec("Data", "Core", "Num Workers", ("data", "num_workers"), "int"),
    FieldSpec("Data", "Core", "Pin Memory", ("data", "pin_memory"), "bool"),
    FieldSpec("Data", "Captions", "Caption Dropout Prob", ("data", "caption_dropout_prob"), "float"),
    FieldSpec("Data", "Captions", "Caption Shuffle Prob", ("data", "caption_shuffle_prob"), "float"),
    FieldSpec("Data", "Captions", "Caption Shuffle Separator", ("data", "caption_shuffle_separator"), "str"),
    FieldSpec("Data", "Captions", "Caption Shuffle Min Tokens", ("data", "caption_shuffle_min_tokens"), "int"),
    FieldSpec("Data", "Buckets", "Bucket Enabled", ("data", "bucket", "enabled"), "bool"),
    FieldSpec("Data", "Buckets", "Bucket Resolutions", ("data", "bucket", "resolutions"), "resolutions"),
    FieldSpec("Data", "Buckets", "Bucket Divisible By", ("data", "bucket", "divisible_by"), "int"),
    FieldSpec("Data", "Buckets", "Bucket Batch Size (optional)", ("data", "bucket", "batch_size"), "int?"),
    FieldSpec("Data", "Buckets", "Bucket Drop Last", ("data", "bucket", "drop_last"), "bool"),
    FieldSpec(
        "Data",
        "Buckets",
        "Per Resolution Batch Sizes (JSON object)",
        ("data", "bucket", "per_resolution_batch_sizes"),
        "json",
    ),
    FieldSpec("Data", "Buckets", "Bucket Log Switches", ("data", "bucket", "log_switches"), "bool"),
    FieldSpec("Data", "Latent Cache", "Latent Cache Enabled", ("data", "latent_cache", "enabled"), "bool"),
    FieldSpec("Data", "Latent Cache", "Latent Cache Dir", ("data", "latent_cache", "cache_dir"), "str", picker="dir"),
    FieldSpec(
        "Data",
        "Latent Cache",
        "Latent Cache DType",
        ("data", "latent_cache", "dtype"),
        "choice",
        ("auto", "fp16", "bf16", "fp32"),
    ),
    FieldSpec(
        "Data",
        "Latent Cache",
        "Latent Cache Build Batch Size",
        ("data", "latent_cache", "build_batch_size"),
        "int",
    ),
    FieldSpec("Eval", "Core", "Eval Backend", ("eval", "backend"), "choice", ("diffusers", "kdiffusion")),
    FieldSpec("Eval", "Core", "Sampler Name", ("eval", "sampler_name")),
    FieldSpec("Eval", "Core", "Scheduler", ("eval", "scheduler")),
    FieldSpec("Eval", "Core", "Inference Steps", ("eval", "num_inference_steps"), "int"),
    FieldSpec("Eval", "Core", "CFG Scale", ("eval", "cfg_scale"), "float"),
    FieldSpec("Eval", "Core", "Prompts Path (optional)", ("eval", "prompts_path"), "str?", picker="file"),
    FieldSpec("Eval", "Core", "Height (optional)", ("eval", "height"), "int?"),
    FieldSpec("Eval", "Core", "Width (optional)", ("eval", "width"), "int?"),
    FieldSpec("Eval", "Core", "Use EMA", ("eval", "use_ema"), "bool"),
    FieldSpec("Eval", "Live", "Live Eval Enabled", ("eval", "live", "enabled"), "bool"),
    FieldSpec("Eval", "Live", "Live Every N Steps (optional)", ("eval", "live", "every_n_steps"), "int?"),
    FieldSpec("Eval", "Live", "Live Max Batches (optional)", ("eval", "live", "max_batches"), "int?"),
    FieldSpec("Eval", "Final", "Final Eval Enabled", ("eval", "final", "enabled"), "bool"),
    FieldSpec("Eval", "Final", "Final Max Batches (optional)", ("eval", "final", "max_batches"), "int?"),
    FieldSpec("Export", "Core", "Save Single File", ("export", "save_single_file"), "bool"),
    FieldSpec("Export", "Core", "Checkpoint Path (optional)", ("export", "checkpoint_path"), "str?", picker="save_file"),
    FieldSpec("Export", "Core", "Converter Script", ("export", "converter_script"), "str", picker="file"),
    FieldSpec("Export", "Core", "Half Precision", ("export", "half_precision"), "bool"),
    FieldSpec("Export", "Core", "Use Safetensors", ("export", "use_safetensors"), "bool"),
    FieldSpec("Export", "Core", "Extra Args (JSON list)", ("export", "extra_args"), "json"),
    FieldSpec("TensorBoard", "Logging", "TensorBoard Enabled", ("training", "tensorboard", "enabled"), "bool"),
    FieldSpec("TensorBoard", "Logging", "TensorBoard Log Dir (optional)", ("training", "tensorboard", "log_dir"), "str?", picker="dir"),
    FieldSpec("TensorBoard", "Logging", "TensorBoard Base Dir", ("training", "tensorboard", "base_dir"), "str", picker="dir"),
    FieldSpec("TensorBoard", "Logging", "Log Grad Norm", ("training", "tensorboard", "log_grad_norm"), "bool"),
    FieldSpec("TensorBoard", "Logging", "Log AMP Scaler", ("training", "tensorboard", "log_scaler"), "bool"),
)


PRESETS = {
    "Smoke Test (50 steps)": {
        ("training", "num_steps"): 50,
        ("training", "num_epochs"): None,
        ("training", "batch_size"): 2,
        ("training", "grad_accum_steps"): 1,
        ("training", "checkpoint_every"): 50,
        ("training", "log_every"): 10,
        ("data", "num_workers"): 4,
        ("eval", "live", "enabled"): False,
        ("eval", "final", "enabled"): False,
    },
    "Balanced (3000 steps)": {
        ("training", "num_steps"): 3000,
        ("training", "num_epochs"): None,
        ("training", "lr_unet"): 5e-6,
        ("training", "lr_text_encoder_1"): 2e-5,
        ("training", "lr_text_encoder_2"): 2e-6,
        ("training", "snr_gamma"): 5.0,
        ("training", "max_grad_norm"): 1.0,
    },
    "UNet Only": {
        ("model", "train_text_encoder_1"): False,
        ("model", "train_text_encoder_2"): False,
        ("training", "lr_text_encoder_1"): None,
        ("training", "lr_text_encoder_2"): None,
        ("training", "te_freeze_fraction"): None,
    },
}


def get_nested(data: dict, path: tuple[str, ...], default=None):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def set_nested(data: dict, path: tuple[str, ...], value):
    cur = data
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


class ScrollableForm(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.inner = ttk.Frame(self.canvas)
        self.inner.columnconfigure(0, weight=1)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)
        self.inner.bind("<Enter>", self._bind_mousewheel)
        self.inner.bind("<Leave>", self._unbind_mousewheel)

    def _on_inner_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _bind_mousewheel(self, _event=None):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        delta = 0
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        elif getattr(event, "delta", 0):
            delta = -1 if event.delta > 0 else 1
        if delta:
            self.canvas.yview_scroll(delta, "units")


class TrainerUIPreview(tk.Tk):
    def __init__(self):
        super().__init__()
        self.repo_root = Path(__file__).resolve().parent
        self.config_path = self.repo_root / "config.json"
        self.config_data: dict = {}

        self.bindings: dict[tuple[str, ...], FieldBinding] = {}
        self.tab_forms: dict[str, ScrollableForm] = {}
        self.section_frames: dict[tuple[str, str], ttk.LabelFrame] = {}
        self.section_rows: dict[tuple[str, str], int] = {}
        self.section_order: dict[str, int] = {}

        self.process: subprocess.Popen | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self._suspend_dirty_tracking = False
        self._dirty = False

        self.status_var = tk.StringVar(value="Idle")
        self.dirty_var = tk.StringVar(value="Saved")
        self.preset_var = tk.StringVar(value=list(PRESETS.keys())[0])
        self.config_path_var = tk.StringVar(value=str(self.config_path))
        self.auto_scroll_var = tk.BooleanVar(value=True)

        self.title("SDXL Trainer UI Preview")
        self.geometry("1420x920")
        self.minsize(1180, 760)

        self._build_style()
        self._build_layout()
        self.reload_config()
        self.after(100, self._poll_logs)

    def _build_style(self):
        style = ttk.Style(self)
        style.configure("Header.TLabel", font=("Noto Sans", 13, "bold"))
        style.configure("Subtle.TLabel", foreground="#5F6773")
        style.configure("Saved.TLabel", foreground="#2A7A2A")
        style.configure("Dirty.TLabel", foreground="#B06A00")
        style.configure("Section.TLabelframe.Label", font=("Noto Sans", 10, "bold"))

    def _build_layout(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=(12, 10))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="SDXL Trainer UI Preview", style="Header.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        ttk.Entry(top, textvariable=self.config_path_var, state="readonly").grid(
            row=0, column=1, sticky="ew", padx=(0, 8)
        )
        ttk.Button(top, text="Reload", command=self.reload_config).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(top, text="Save", command=self.save_config).grid(row=0, column=3)

        preset_row = ttk.Frame(top)
        preset_row.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        preset_row.columnconfigure(7, weight=1)

        ttk.Label(preset_row, text="Preset:", style="Subtle.TLabel").grid(row=0, column=0, padx=(0, 6))
        ttk.Combobox(
            preset_row,
            textvariable=self.preset_var,
            values=list(PRESETS.keys()),
            state="readonly",
            width=26,
        ).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(preset_row, text="Apply Preset", command=self.apply_preset).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(preset_row, text="Load Defaults", command=self.load_defaults).grid(row=0, column=3, padx=(0, 8))
        ttk.Button(preset_row, text="Open Output Dir", command=self.open_output_dir).grid(row=0, column=4, padx=(0, 8))
        ttk.Button(preset_row, text="Launch TensorBoard", command=self.launch_tensorboard).grid(
            row=0, column=5, padx=(0, 8)
        )
        self.dirty_label = ttk.Label(preset_row, textvariable=self.dirty_var, style="Saved.TLabel")
        self.dirty_label.grid(row=0, column=6, padx=(0, 8))
        ttk.Label(
            preset_row,
            text="Edits only config + process control. Training script remains unchanged.",
            style="Subtle.TLabel",
        ).grid(row=0, column=7, sticky="e")

        body = ttk.Panedwindow(self, orient=tk.VERTICAL)
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        form_wrap = ttk.Frame(body)
        form_wrap.columnconfigure(0, weight=1)
        form_wrap.rowconfigure(0, weight=1)
        body.add(form_wrap, weight=4)

        notebook = ttk.Notebook(form_wrap)
        notebook.grid(row=0, column=0, sticky="nsew")

        for tab_name in self._tab_order():
            form = ScrollableForm(notebook)
            notebook.add(form, text=tab_name)
            self.tab_forms[tab_name] = form
            self.section_order[tab_name] = 0

        for spec in FIELD_SPECS:
            self._add_field(spec)

        log_wrap = ttk.Frame(body, padding=(0, 8, 0, 0))
        log_wrap.columnconfigure(0, weight=1)
        log_wrap.rowconfigure(1, weight=1)
        body.add(log_wrap, weight=2)

        controls = ttk.Frame(log_wrap)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(5, weight=1)

        self.start_button = ttk.Button(controls, text="Start Training", command=self.start_training)
        self.start_button.grid(row=0, column=0, padx=(0, 6))
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop_training, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(0, 6))
        ttk.Button(controls, text="Clear Log", command=self.clear_log).grid(row=0, column=2, padx=(0, 6))
        ttk.Checkbutton(controls, text="Auto Scroll", variable=self.auto_scroll_var).grid(
            row=0, column=3, padx=(0, 6), sticky="w"
        )
        ttk.Label(controls, text="Status:", style="Subtle.TLabel").grid(row=0, column=4, sticky="e", padx=(0, 4))
        ttk.Label(controls, textvariable=self.status_var).grid(row=0, column=5, sticky="w")

        log_frame = ttk.LabelFrame(log_wrap, text="Trainer Log", padding=8, style="Section.TLabelframe")
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap="none", height=14, font=("monospace", 10))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(log_frame, orient=tk.HORIZONTAL, command=self.log_text.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.log_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _tab_order(self):
        seen = []
        for spec in FIELD_SPECS:
            if spec.tab not in seen:
                seen.append(spec.tab)
        return seen

    def _ensure_section(self, tab: str, section: str):
        key = (tab, section)
        if key in self.section_frames:
            return self.section_frames[key]

        parent = self.tab_forms[tab].inner
        section_frame = ttk.LabelFrame(parent, text=section, padding=10, style="Section.TLabelframe")
        section_frame.grid(row=self.section_order[tab], column=0, sticky="ew", pady=(0, 10))
        section_frame.columnconfigure(0, weight=1)
        self.section_order[tab] += 1

        self.section_frames[key] = section_frame
        self.section_rows[key] = 0
        return section_frame

    def _add_field(self, spec: FieldSpec):
        section_frame = self._ensure_section(spec.tab, spec.section)
        row_index = self.section_rows[(spec.tab, spec.section)]
        row = ttk.Frame(section_frame)
        row.grid(row=row_index, column=0, sticky="ew", pady=3)
        row.columnconfigure(1, weight=1)

        ttk.Label(row, text=spec.label, width=36).grid(row=0, column=0, sticky="w", padx=(0, 10))

        if spec.kind == "bool":
            var: tk.Variable = tk.BooleanVar(value=False)
            widget = ttk.Checkbutton(row, variable=var)
            widget.grid(row=0, column=1, sticky="w")
        elif spec.kind == "choice":
            var = tk.StringVar()
            widget = ttk.Combobox(row, textvariable=var, values=spec.options, state="readonly")
            widget.grid(row=0, column=1, sticky="ew")
        else:
            var = tk.StringVar()
            widget = ttk.Entry(row, textvariable=var)
            widget.grid(row=0, column=1, sticky="ew")

        if spec.picker:
            ttk.Button(row, text="...", width=3, command=lambda p=spec.path: self._browse_for_field(p)).grid(
                row=0, column=2, padx=(6, 0)
            )

        if spec.kind.endswith("?"):
            ttk.Button(row, text="Clear", width=6, command=lambda p=spec.path: self._clear_optional(p)).grid(
                row=0, column=3, padx=(6, 0)
            )

        var.trace_add("write", lambda *_args: self._on_any_field_changed())
        self.bindings[spec.path] = FieldBinding(var=var, spec=spec, row=row)
        self.section_rows[(spec.tab, spec.section)] += 1

    def _on_any_field_changed(self):
        if self._suspend_dirty_tracking:
            return
        self._set_dirty(True)

    def _set_dirty(self, dirty: bool):
        self._dirty = bool(dirty)
        if self._dirty:
            self.dirty_var.set("Unsaved changes")
            self.dirty_label.configure(style="Dirty.TLabel")
        else:
            self.dirty_var.set("Saved")
            self.dirty_label.configure(style="Saved.TLabel")

    def _format_value_for_var(self, spec: FieldSpec, value):
        if spec.kind == "bool":
            return bool(value)
        if spec.kind == "choice":
            if value is None:
                return "none" if "none" in spec.options else ""
            return str(value)
        if spec.kind == "resolutions":
            if not value:
                return ""
            return ", ".join(f"{int(w)}x{int(h)}" for w, h in value)
        if spec.kind == "json":
            if value is None:
                return ""
            return json.dumps(value)
        return "" if value is None else str(value)

    def _parse_field_value(self, raw_value, spec: FieldSpec):
        kind = spec.kind
        if kind == "bool":
            return bool(raw_value)

        text = str(raw_value).strip()

        if kind == "str":
            return text
        if kind == "str?":
            return text if text else None
        if kind == "int":
            if not text:
                raise ValueError("expected integer")
            return int(text)
        if kind == "int?":
            return None if not text else int(text)
        if kind == "float":
            if not text:
                raise ValueError("expected float")
            return float(text)
        if kind == "float?":
            return None if not text else float(text)
        if kind == "choice":
            if text == "none":
                return None
            if spec.options and text not in spec.options:
                raise ValueError(f"expected one of: {', '.join(spec.options)}")
            return text
        if kind == "resolutions":
            if not text:
                return []
            parsed = []
            chunks = [token.strip() for token in text.replace("\n", ",").split(",") if token.strip()]
            for token in chunks:
                parts = token.lower().split("x")
                if len(parts) != 2:
                    raise ValueError("use format like 1024x1024, 832x1216")
                width = int(parts[0].strip())
                height = int(parts[1].strip())
                if width <= 0 or height <= 0:
                    raise ValueError("resolution values must be > 0")
                parsed.append([width, height])
            return parsed
        if kind == "json":
            if not text:
                return None
            return json.loads(text)

        raise ValueError(f"unsupported field kind: {kind}")

    def _collect_updated_config(self):
        updated = json.loads(json.dumps(self.config_data))
        errors = []

        for path, binding in self.bindings.items():
            try:
                value = self._parse_field_value(binding.var.get(), binding.spec)
                set_nested(updated, path, value)
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{binding.spec.label}: {exc}")

        errors.extend(self._validate_rules(updated))
        return updated, errors

    def _validate_rules(self, cfg: dict):
        errors = []
        model_cfg = cfg.get("model", {}) or {}
        training_cfg = cfg.get("training", {}) or {}

        if training_cfg.get("num_steps") is None and training_cfg.get("num_epochs") is None:
            errors.append("Set either training.num_steps or training.num_epochs.")

        if model_cfg.get("train_text_encoder_1") and training_cfg.get("lr_text_encoder_1") is None:
            errors.append("train_text_encoder_1=true requires training.lr_text_encoder_1.")

        if model_cfg.get("train_text_encoder_2") and training_cfg.get("lr_text_encoder_2") is None:
            errors.append("train_text_encoder_2=true requires training.lr_text_encoder_2.")

        return errors

    def reload_config(self):
        cfg = load_config(self.config_path)
        self.config_data = cfg

        self._suspend_dirty_tracking = True
        try:
            for path, binding in self.bindings.items():
                value = get_nested(cfg, path)
                binding.var.set(self._format_value_for_var(binding.spec, value))
        finally:
            self._suspend_dirty_tracking = False

        self._set_dirty(False)
        self.status_var.set("Config loaded")
        self._append_log(f"[ui] loaded config: {self.config_path}")

    def save_config(self, show_message: bool = True):
        updated, errors = self._collect_updated_config()
        if errors:
            messagebox.showerror("Invalid Values", "\n".join(errors[:20]))
            self.status_var.set("Validation failed")
            return False

        self.config_data = updated
        text = json.dumps(updated, indent=2)
        self.config_path.write_text(text + "\n", encoding="utf-8")
        self._set_dirty(False)
        self.status_var.set("Config saved")
        self._append_log(f"[ui] saved config: {self.config_path}")
        if show_message:
            messagebox.showinfo("Saved", "Config saved successfully.")
        return True

    def _clear_optional(self, path: tuple[str, ...]):
        binding = self.bindings.get(path)
        if binding is None:
            return
        if binding.spec.kind == "choice" and "none" in binding.spec.options:
            binding.var.set("none")
        else:
            binding.var.set("")

    def _browse_for_field(self, path: tuple[str, ...]):
        binding = self.bindings.get(path)
        if binding is None or not binding.spec.picker:
            return

        current = str(binding.var.get()).strip()
        initial = self.repo_root
        if current:
            candidate = Path(current).expanduser()
            if candidate.exists():
                initial = candidate if candidate.is_dir() else candidate.parent
            else:
                initial = candidate.parent if candidate.parent.exists() else self.repo_root

        selected = None
        if binding.spec.picker == "dir":
            selected = filedialog.askdirectory(initialdir=str(initial))
        elif binding.spec.picker == "file":
            selected = filedialog.askopenfilename(initialdir=str(initial))
        elif binding.spec.picker == "save_file":
            selected = filedialog.asksaveasfilename(initialdir=str(initial))

        if selected:
            binding.var.set(selected)

    def apply_preset(self):
        name = self.preset_var.get()
        preset = PRESETS.get(name)
        if not preset:
            return

        if not messagebox.askyesno("Apply Preset", f"Apply preset '{name}' to current form values?"):
            return

        self._suspend_dirty_tracking = True
        try:
            for path, value in preset.items():
                binding = self.bindings.get(path)
                if binding is None:
                    continue
                binding.var.set(self._format_value_for_var(binding.spec, value))
        finally:
            self._suspend_dirty_tracking = False

        self._set_dirty(True)
        self.status_var.set(f"Preset applied: {name}")
        self._append_log(f"[ui] preset applied: {name}")

    def load_defaults(self):
        if not messagebox.askyesno("Load Defaults", "Load DEFAULT_CONFIG values into the form (not saved yet)?"):
            return

        self.config_data = json.loads(json.dumps(DEFAULT_CONFIG))
        self._suspend_dirty_tracking = True
        try:
            for path, binding in self.bindings.items():
                value = get_nested(self.config_data, path)
                binding.var.set(self._format_value_for_var(binding.spec, value))
        finally:
            self._suspend_dirty_tracking = False

        self._set_dirty(True)
        self.status_var.set("Defaults loaded (unsaved)")
        self._append_log("[ui] loaded defaults into form")

    def _resolve_output_dir(self, cfg: dict):
        training_cfg = cfg.get("training", {}) or {}
        run_cfg = cfg.get("run", {}) or {}

        output_dir = training_cfg.get("output_dir")
        if output_dir:
            return Path(output_dir).expanduser()

        run_name = run_cfg.get("name") or "run"
        output_root = Path(run_cfg.get("output_root", ".output")).expanduser()
        return output_root / run_name

    def open_output_dir(self):
        updated, errors = self._collect_updated_config()
        if errors:
            messagebox.showerror("Invalid Values", "\n".join(errors[:20]))
            return

        output_dir = self._resolve_output_dir(updated)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.Popen(["xdg-open", str(output_dir)], cwd=self.repo_root)
            self._append_log(f"[ui] opened output directory: {output_dir}")
        except FileNotFoundError:
            messagebox.showinfo("Output Directory", str(output_dir))

    def launch_tensorboard(self):
        updated, errors = self._collect_updated_config()
        if errors:
            messagebox.showerror("Invalid Values", "\n".join(errors[:20]))
            return

        tb_cfg = get_nested(updated, ("training", "tensorboard"), {}) or {}
        run_name = get_nested(updated, ("run", "name")) or "run"
        explicit_log_dir = tb_cfg.get("log_dir")
        if explicit_log_dir:
            log_dir = Path(explicit_log_dir).expanduser()
        else:
            base_dir = Path(tb_cfg.get("base_dir", "./logs/tensorboard")).expanduser()
            log_dir = base_dir / run_name
        log_dir.mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, "-m", "tensorboard", "--logdir", str(log_dir)]
        try:
            subprocess.Popen(cmd, cwd=self.repo_root)
            self._append_log(f"[ui] launched tensorboard: {' '.join(cmd)}")
        except Exception as exc:
            messagebox.showerror("TensorBoard", str(exc))

    def start_training(self):
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("Already Running", "Training process is already running.")
            return

        if not self.save_config(show_message=False):
            return

        cmd = [sys.executable, "train_sdxl_ff.py"]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self._append_log("[ui] starting: " + " ".join(cmd))

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as exc:
            self.status_var.set("Start failed")
            messagebox.showerror("Start Failed", str(exc))
            return

        self.status_var.set("Running")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        threading.Thread(target=self._stream_process_output, daemon=True).start()

    def stop_training(self):
        if self.process is None or self.process.poll() is not None:
            self.status_var.set("Idle")
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            return

        try:
            if self.status_var.get() == "Stopping":
                self.process.terminate()
                self._append_log("[ui] forced terminate requested")
            else:
                self.process.send_signal(signal.SIGINT)
                self._append_log("[ui] stop requested (SIGINT)")
            self.status_var.set("Stopping")
        except Exception as exc:
            self._append_log(f"[ui] failed to stop process: {exc}")

    def _stream_process_output(self):
        proc = self.process
        if proc is None:
            return

        if proc.stdout is not None:
            for line in proc.stdout:
                self.log_queue.put(line.rstrip("\n"))

        return_code = proc.wait()
        self.log_queue.put(f"[ui] process exited with code {return_code}")
        self.log_queue.put("__PROCESS_FINISHED__")

    def _poll_logs(self):
        while True:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break

            if item == "__PROCESS_FINISHED__":
                self.status_var.set("Idle")
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.process = None
            else:
                self._append_log(item)

        self.after(100, self._poll_logs)

    def clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def _append_log(self, text: str):
        self.log_text.insert(tk.END, text + "\n")
        if self.auto_scroll_var.get():
            self.log_text.see(tk.END)

    def _on_close(self):
        if self.process is not None and self.process.poll() is None:
            should_close = messagebox.askyesno(
                "Training Running",
                "A training process is still running. Close window and send SIGINT?",
            )
            if not should_close:
                return
            self.stop_training()
        self.destroy()


def main():
    app = TrainerUIPreview()
    app.mainloop()


if __name__ == "__main__":
    main()
