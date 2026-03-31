from __future__ import annotations

import json
import random
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from firefighting_simclr.data import (
    SimCLRViewTransform,
    SymbolCropDataset,
    build_label_mapping,
    build_supervised_transform,
    filter_records,
    load_split_records,
)
from firefighting_simclr.models import (
    ClassificationModel,
    SimCLRModel,
    build_backbone,
    load_encoder_state_dict,
    nt_xent_loss,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(preferred: str) -> torch.device:
    if preferred == "auto":
        preferred = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(preferred)


def make_data_loader(
    dataset,
    batch_size: int,
    shuffle: bool = False,
    sampler: WeightedRandomSampler | None = None,
    num_workers: int = 0,
    device: torch.device | None = None,
    drop_last: bool = False,
) -> DataLoader:
    pin_memory = device is not None and device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def _autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return nullcontext()


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    def _default(value):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")

# The pretraining aims to learn a good feature representation of the crops using contrastive learning,
# where we apply strong data augmentations to create two different views of the same crop and train
# the model to bring their projections closer together while pushing apart projections of different crops.
def run_pretraining(args) -> Path:
    seed_everything(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records, _ = load_split_records(args.data_dir, "train")
    
    # build a dataset that applies the SimCLRViewTransform to each crop,
    # which generates two augmented views of the same crop for contrastive learning
    dataset = SymbolCropDataset(
        train_records,
        transform=SimCLRViewTransform(args.image_size),
        padding_ratio=args.crop_padding,
    )
    loader = make_data_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        drop_last=True,
    )

    encoder, feature_dim = build_backbone(args.backbone)
    model = SimCLRModel(
        encoder=encoder,
        feature_dim=feature_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dim=args.projection_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.pretrain_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.simclr_epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")

    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, args.simclr_epochs + 1):
        running_loss = 0.0
        progress = tqdm(loader, desc=f"SimCLR epoch {epoch}/{args.simclr_epochs}", leave=False)

        for view_a, view_b in progress:
            view_a = view_a.to(device, non_blocking=True)
            view_b = view_b.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, enabled=args.amp and device.type == "cuda"):
                projection_a, projection_b = model(view_a, view_b)
                loss = nt_xent_loss(projection_a, projection_b, args.temperature)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * view_a.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        epoch_loss = running_loss / len(dataset)
        history.append({"epoch": epoch, "loss": epoch_loss})

    checkpoint_path = output_dir / "simclr_pretrain.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone": args.backbone,
            "feature_dim": feature_dim,
            "projection_dim": args.projection_dim,
            "projection_hidden_dim": args.projection_hidden_dim,
            "image_size": args.image_size,
            "crop_padding": args.crop_padding,
        },
        checkpoint_path,
    )
    _save_json(
        output_dir / "pretrain_history.json",
        {
            "device": str(device),
            "num_crops": len(dataset),
            "epochs": history,
        },
    )
    return checkpoint_path


def _build_balanced_sampler(records, label_mapping):
    class_counts = Counter(label_mapping[record.category_id] for record in records)
    sample_weights = [1.0 / class_counts[label_mapping[record.category_id]] for record in records]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def _evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    average_loss = float(total_loss / max(1, len(targets)))
    accuracy = float(accuracy_score(targets, predictions))
    macro_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_samples": int(len(targets)),
        "report": report,
    }


def run_finetuning(args, encoder_checkpoint: str | None = None) -> Path:
    seed_everything(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_mapping, class_names = build_label_mapping(args.data_dir)
    train_records, _ = load_split_records(args.data_dir, "train")
    valid_records, _ = load_split_records(args.data_dir, "valid")
    test_records, _ = load_split_records(args.data_dir, "test")

    valid_total = len(valid_records)
    test_total = len(test_records)
    train_records = filter_records(train_records, label_mapping)
    valid_records = filter_records(valid_records, label_mapping)
    test_records = filter_records(test_records, label_mapping)

    train_dataset = SymbolCropDataset(
        train_records,
        transform=build_supervised_transform(args.image_size, train=True),
        label_mapping=label_mapping,
        padding_ratio=args.crop_padding,
    )
    valid_dataset = SymbolCropDataset(
        valid_records,
        transform=build_supervised_transform(args.image_size, train=False),
        label_mapping=label_mapping,
        padding_ratio=args.crop_padding,
    )
    test_dataset = SymbolCropDataset(
        test_records,
        transform=build_supervised_transform(args.image_size, train=False),
        label_mapping=label_mapping,
        padding_ratio=args.crop_padding,
    )

    train_loader = make_data_loader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=_build_balanced_sampler(train_records, label_mapping),
        num_workers=args.num_workers,
        device=device,
    )
    valid_loader = make_data_loader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    test_loader = make_data_loader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    encoder, feature_dim = build_backbone(args.backbone)
    if encoder_checkpoint:
        # load the pretrained encoder weights from the SimCLR pretraining checkpoint
        state_dict = load_encoder_state_dict(encoder_checkpoint, device)
        encoder.load_state_dict(state_dict, strict=True)

    model = ClassificationModel(
        encoder=encoder,
        feature_dim=feature_dim,
        num_classes=len(class_names),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")

    history: list[dict[str, float]] = []
    best_macro_f1 = float("-inf")
    best_path = output_dir / "best_symbol_classifier.pt"

    for epoch in range(1, args.finetune_epochs + 1):
        encoder_trainable = epoch > args.linear_probe_epochs
        for parameter in model.encoder.parameters():
            parameter.requires_grad = encoder_trainable

        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Fine-tune epoch {epoch}/{args.finetune_epochs}", leave=False)

        for inputs, labels in progress:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, enabled=args.amp and device.type == "cuda"):
                logits = model(inputs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / len(train_dataset)
        valid_metrics = _evaluate_classifier(model, valid_loader, criterion, device, class_names)
        epoch_summary = {
            "epoch": epoch,
            "encoder_trainable": encoder_trainable,
            "train_loss": train_loss,
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"],
        }
        history.append(epoch_summary)

        if valid_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = valid_metrics["macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "backbone": args.backbone,
                    "feature_dim": feature_dim,
                    "class_names": class_names,
                    "label_mapping": label_mapping,
                    "image_size": args.image_size,
                    "crop_padding": args.crop_padding,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    test_metrics = _evaluate_classifier(model, test_loader, criterion, device, class_names)

    _save_json(
        output_dir / "dataset_summary.json",
        {
            "train_crops": len(train_dataset),
            "valid_crops_used": len(valid_dataset),
            "valid_crops_ignored": valid_total - len(valid_dataset),
            "test_crops_used": len(test_dataset),
            "test_crops_ignored": test_total - len(test_dataset),
            "num_train_classes": len(class_names),
            "class_names": class_names,
        },
    )
    _save_json(output_dir / "finetune_history.json", {"epochs": history})
    _save_json(
        output_dir / "test_metrics.json",
        {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "num_samples": test_metrics["num_samples"],
            "report": test_metrics["report"],
        },
    )
    return best_path
