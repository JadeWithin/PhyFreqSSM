from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "phyfreqssm_main",
        "output_root": "outputs",
        "save_train_indices": True,
        "save_predictions": True,
    },
    "dataset": {
        "name": "IndianPines",
        "root": "../data",
        "split_mode": "random_ratio",
        "train_ratio": 0.05,
        "val_ratio": 0.05,
        "shots_per_class": 5,
        "block_size": 8,
        "spatial_guard_band": 0,
        "spatial_axis": "both",
        "num_workers": 4,
    },
    "preprocessing": {
        "standardize": True,
        "pca_bands": None,
        "patch_size": 13,
        "spectral_smoothing": False,
        "smoothing_kernel": 5,
        "class_balanced_sampler": False,
        "compression_ratio": 0.4,
        "entropy_guided_sampling": True,
        "train_augmentation": True,
        "aug_rot90": True,
        "aug_hflip": True,
        "aug_vflip": True,
    },
    "model": {
        "name": "phyfreqssm",
        "in_channels": None,
        "num_classes": None,
        "embed_dim": 128,
        "depth": 4,
        "dropout": 0.1,
        "d_state": 64,
        "scan_mode": "single",
        "use_dsst": True,
        "use_spatial_freq_stem": True,
        "use_raster_tokenizer": True,
        "use_fg_ssd": True,
        "use_local_branch": False,
        "use_mamba_branch": True,
        "use_pch": True,
        "use_frequency_conditioning": True,
        "use_lowfreq_for_delta_b": True,
        "use_highfreq_for_c": True,
        "use_noncausal_single_pass": True,
        "use_multiscan_baseline": False,
        "use_plain_reconstruction_head": False,
        "allow_mamba_fallback": False,
        "dsst_version": "full",
        "fgssd_variant": "full",
    },
    "loss": {
        "label_smoothing": 0.0,
        "reconstruction_type": "mse",
        "physics_type": "sam_smoothness",
        "lambda_recon": 0.15,
        "lambda_physics": 0.05,
        "stage_a_epochs": 20,
        "stage_a_recon_weight": 0.0,
    },
    "train": {
        "device": "cuda",
        "optimizer": "adamw",
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "epochs": 250,
        "warmup_epochs": 10,
        "amp": True,
        "grad_clip": 1.0,
        "early_stopping_patience": 40,
        "seed": 3407,
        "seeds": [3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416],
        "deterministic": True,
        "metric_for_best": "hybrid",
        "metric_weights": {"oa": 0.7, "macro_f1": 0.3},
        "report_confidence_interval": True,
    },
    "benchmark": {
        "num_warmup": 20,
        "num_iters": 100,
    },
    "logging": {
        "print_freq": 20,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    candidates = [path]
    repo_root = Path(__file__).resolve().parents[1]

    if not path.is_absolute():
        candidates.extend(
            [
                Path.cwd() / path,
                repo_root / path,
                Path.cwd() / "configs" / path,
                repo_root / "configs" / path,
            ]
        )

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return path


def load_config(config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is not None:
        resolved_path = resolve_config_path(config_path)
        with resolved_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        deep_update(config, loaded)
    if overrides:
        deep_update(config, overrides)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=False)
