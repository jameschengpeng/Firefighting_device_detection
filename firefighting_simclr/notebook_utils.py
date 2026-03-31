from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from firefighting_simclr.training import run_finetuning, run_pretraining


def find_repo_root(start: Path | str | None = None) -> Path:
    current = Path.cwd() if start is None else Path(start)
    current = current.resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "Data").exists() and (candidate / "firefighting_simclr").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find the repository root. Expected folders named 'Data' and "
        "'firefighting_simclr' in the current directory or one of its parents."
    )


def ensure_repo_on_path(start: Path | str | None = None) -> Path:
    root = find_repo_root(start)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def make_experiment_args(
    *,
    data_dir: Path | str,
    output_dir: Path | str,
    backbone: str = "resnet18",
    image_size: int = 96,
    batch_size: int = 128,
    num_workers: int = 0,
    crop_padding: float = 0.15,
    device: str = "auto",
    seed: int = 42,
    weight_decay: float = 1e-4,
    amp: bool = True,
    simclr_epochs: int = 30,
    pretrain_lr: float = 3e-4,
    temperature: float = 0.2,
    projection_hidden_dim: int = 256,
    projection_dim: int = 128,
    finetune_epochs: int = 25,
    finetune_lr: float = 1e-3,
    dropout: float = 0.2,
    label_smoothing: float = 0.05,
    linear_probe_epochs: int = 5,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        backbone=backbone,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        crop_padding=crop_padding,
        device=device,
        seed=seed,
        weight_decay=weight_decay,
        amp=amp,
        simclr_epochs=simclr_epochs,
        pretrain_lr=pretrain_lr,
        temperature=temperature,
        projection_hidden_dim=projection_hidden_dim,
        projection_dim=projection_dim,
        finetune_epochs=finetune_epochs,
        finetune_lr=finetune_lr,
        dropout=dropout,
        label_smoothing=label_smoothing,
        linear_probe_epochs=linear_probe_epochs,
    )


def clone_args(args: SimpleNamespace, **overrides: Any) -> SimpleNamespace:
    payload = vars(args).copy()
    payload.update(overrides)
    return SimpleNamespace(**payload)


def run_full_pipeline(args: SimpleNamespace) -> dict[str, Path]:
    base_output_dir = Path(args.output_dir)
    pretrain_args = clone_args(args, output_dir=base_output_dir / "pretrain")
    finetune_args = clone_args(args, output_dir=base_output_dir / "finetune")

    encoder_checkpoint = run_pretraining(pretrain_args)
    classifier_checkpoint = run_finetuning(
        finetune_args,
        encoder_checkpoint=str(encoder_checkpoint),
    )

    return {
        "pretrain_dir": pretrain_args.output_dir,
        "finetune_dir": finetune_args.output_dir,
        "encoder_checkpoint": encoder_checkpoint,
        "classifier_checkpoint": classifier_checkpoint,
    }


def load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_test_metrics(output_dir: Path | str) -> dict[str, Any]:
    output_dir = Path(output_dir)
    return load_json(output_dir / "finetune" / "test_metrics.json")
