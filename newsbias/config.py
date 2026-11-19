"""Run configuration, read from a TOML file in configs/.

A config may start with ``extends = "base.toml"`` (path relative to the file) and override any key. Unknown keys
are an error, so a typo cannot silently fall back to a default.
"""
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    # data
    data_path: str = "../news-bias-dataset/data/annotated/art_A1.jsonl"
    # model
    model_name: str = "ai4bharat/IndicBERTv2-MLM-only"
    max_length: int = 256
    head_dropout: float = 0.2
    freeze_embeddings: bool = True
    freeze_layers: int = 0
    # optimisation (fixed before evaluation; never tuned on held-out folds)
    epochs: int = 8
    batch_size: int = 8
    lr: float = 3e-5
    head_lr: float = 1e-3
    layer_decay: float = 0.9
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    loss_weights: dict = field(default_factory=lambda: {"political": 1.0, "gender": 1.0})
    # class imbalance
    class_weight_power: float = 1.0
    confidence_weighting: bool = False
    # shortcut control
    role_mask_p: float = 0.5
    gender_swap_neutral: bool = True
    # cross-validation and reporting
    folds: int = 5
    fold_seed: int = 13
    seeds: list = field(default_factory=lambda: [13])
    group_threshold: float = 0.5
    bootstrap: int = 1000
    explain_eval: bool = False
    # explanations
    ig_steps: int = 16
    top_phrases: int = 5
    # runtime
    cpu_bf16: bool = False
    threads: int = 0
    eval_batch_size: int = 16

    def to_dict(self):
        return asdict(self)


def _read(path, depth=0):
    path = Path(path)
    if depth > 5:
        raise ValueError("config 'extends' chain is too deep")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    parent = raw.pop("extends", None)
    merged = _read(path.parent / parent, depth + 1) if parent else {}
    merged.update(raw)
    return merged


def load_config(path=None, **overrides):
    raw = _read(path) if path else {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(sorted(unknown))}")
    return Config(**raw)
