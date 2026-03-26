"""Train learned spread model v2 (spatial U-Net + ONNX export)."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from ml.spread.hindcast_dataset import (
    V2_TENSOR_CHANNELS,
    build_hindcast_tensor_dataset,
)
from ml.spread.runtime_contract import CANONICAL_CHANNEL_METADATA, SpreadRuntimeContract, write_contract

LOGGER = logging.getLogger("train_spread_v2")


def _maybe_git_sha() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip() or None
    except Exception:
        return None


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        depth: int = 4,
        base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.depth = depth
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        enc_channels = []
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.down_blocks.append(ConvBlock(ch, out_ch, dropout=dropout))
            self.pools.append(nn.MaxPool2d(kernel_size=2))
            enc_channels.append(out_ch)
            ch = out_ch

        self.bottleneck = ConvBlock(ch, ch * 2, dropout=dropout)
        ch = ch * 2

        self.up_transpose = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for skip_ch in reversed(enc_channels):
            self.up_transpose.append(nn.ConvTranspose2d(ch, skip_ch, kernel_size=2, stride=2))
            self.up_blocks.append(ConvBlock(skip_ch * 2, skip_ch, dropout=dropout))
            ch = skip_ch

        self.head = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        cur = x
        for block, pool in zip(self.down_blocks, self.pools):
            cur = block(cur)
            skips.append(cur)
            cur = pool(cur)

        cur = self.bottleneck(cur)
        for up_t, up_b, skip in zip(self.up_transpose, self.up_blocks, reversed(skips)):
            cur = up_t(cur)
            if cur.shape[-2:] != skip.shape[-2:]:
                cur = F.interpolate(cur, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            cur = torch.cat([cur, skip], dim=1)
            cur = up_b(cur)
        return self.head(cur)


@dataclass(slots=True)
class Sample:
    ref_time: datetime
    region_bucket: int
    x: np.ndarray  # (C,H,W)
    y: np.ndarray  # (T,H,W)


def _group_cases(cases: list[dict[str, Any]], horizons: list[int]) -> list[Sample]:
    by_ref: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    for case in cases:
        ref = str(case["ref_time"])
        bucket = int(case.get("region_bucket", 0))
        key = (ref, bucket)
        by_ref.setdefault(key, {})[int(case["horizon_h"])] = case

    samples: list[Sample] = []
    for (ref_str, bucket), payload in by_ref.items():
        if any(h not in payload for h in horizons):
            continue
        base_case = payload[horizons[0]]
        y_stack = np.stack([payload[h]["y_tensor"] for h in horizons], axis=0).astype(np.float32)
        samples.append(
            Sample(
                ref_time=pd_to_datetime_utc(ref_str),
                region_bucket=bucket,
                x=np.asarray(base_case["x_tensor"], dtype=np.float32),
                y=y_stack,
            )
        )
    return samples


def pd_to_datetime_utc(value: Any) -> datetime:
    import pandas as pd

    ts = pd.to_datetime(value, utc=True)
    return ts.to_pydatetime().astimezone(timezone.utc)


def _split_samples(
    samples: list[Sample],
    *,
    holdout_year: int | None,
    validation_region_buckets: set[int],
) -> tuple[list[Sample], list[Sample]]:
    if not samples:
        return [], []

    if holdout_year is None:
        holdout_year = max(s.ref_time.year for s in samples)
    train = [
        s
        for s in samples
        if not (s.ref_time.year >= holdout_year and s.region_bucket in validation_region_buckets)
    ]
    eval_set = [
        s
        for s in samples
        if s.ref_time.year >= holdout_year and s.region_bucket in validation_region_buckets
    ]
    if not train or not eval_set:
        ordered = sorted(samples, key=lambda s: s.ref_time)
        idx = max(1, int(len(ordered) * 0.8))
        train = ordered[:idx]
        eval_set = ordered[idx:]
    return train, eval_set


class TensorCaseDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        return torch.from_numpy(sample.x), torch.from_numpy(sample.y)


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = torch.sum(probs * target, dim=(2, 3))
    union = torch.sum(probs, dim=(2, 3)) + torch.sum(target, dim=(2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def _weighted_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_weight: float,
    horizon_weights: torch.Tensor,
    bce_weight: float,
    dice_weight: float,
) -> torch.Tensor:
    pw = torch.tensor([pos_weight], dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
    bce = (bce * horizon_weights.view(1, -1, 1, 1)).mean()
    dice = _dice_loss(logits, target)
    return bce_weight * bce + dice_weight * dice


def _evaluate_brier(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[float]]:
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            probs = torch.sigmoid(model(x))
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    if not all_probs:
        return float("inf"), []
    p = np.concatenate(all_probs, axis=0)
    t = np.concatenate(all_targets, axis=0)
    per_h = [float(np.mean((p[:, i] - t[:, i]) ** 2)) for i in range(p.shape[1])]
    return float(np.mean(per_h)), per_h


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_spread_v2(config: dict[str, Any]) -> Path:
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    horizons = [int(h) for h in config.get("horizons_hours", [24, 48, 72])]
    channel_names = tuple(config.get("channels", list(V2_TENSOR_CHANNELS)))

    start_time = datetime.fromisoformat(str(config["start_time"])).replace(tzinfo=timezone.utc)
    end_time = datetime.fromisoformat(str(config["end_time"])).replace(tzinfo=timezone.utc)
    raw_negative_ratio = config.get("negative_ratio", 3.0)
    negative_ratio = None if raw_negative_ratio is None else float(raw_negative_ratio)

    cases = build_hindcast_tensor_dataset(
        region_name=str(config["region_name"]),
        bbox=tuple(float(v) for v in config["bbox"]),
        start_time=start_time,
        end_time=end_time,
        horizons_hours=horizons,
        min_detections=int(config.get("min_detections", 5)),
        interval_hours=int(config.get("interval_hours", 24)),
        negative_ratio=negative_ratio,
        min_negative_samples=int(config.get("min_negative_samples", 500)),
        seed=seed,
        tensor_channels=channel_names,
    )
    if not cases:
        raise ValueError("No tensor cases were generated for v2 training.")

    samples = _group_cases(cases, horizons)
    if not samples:
        raise ValueError("No complete ref_time groups found with all configured horizons.")

    split_cfg = config.get("split", {}) or {}
    holdout_year = split_cfg.get("holdout_year")
    val_buckets = set(int(v) for v in split_cfg.get("validation_region_buckets", [0]))
    train_samples, eval_samples = _split_samples(
        samples,
        holdout_year=int(holdout_year) if holdout_year is not None else None,
        validation_region_buckets=val_buckets,
    )
    if not train_samples or not eval_samples:
        raise ValueError("Failed to produce non-empty train/eval sets.")

    train_cfg = config.get("training", {}) or {}
    model_cfg = config.get("model", {}) or {}
    export_cfg = config.get("export", {}) or {}

    batch_size = int(train_cfg.get("batch_size", 4))
    epochs = int(train_cfg.get("epochs", 60))
    lr = float(train_cfg.get("learning_rate", 1e-3))
    wd = float(train_cfg.get("weight_decay", 1e-4))
    patience = int(train_cfg.get("early_stopping_patience", 10))
    pos_weight = float(train_cfg.get("pos_weight", 8.0))
    bce_weight = float(train_cfg.get("bce_weight", 0.6))
    dice_weight = float(train_cfg.get("dice_weight", 0.4))
    horizon_weights = torch.tensor(
        train_cfg.get("horizon_weights", [0.5, 0.3, 0.2]),
        dtype=torch.float32,
    )
    if horizon_weights.numel() != len(horizons):
        raise ValueError("training.horizon_weights must match horizons_hours length.")

    device = torch.device("cpu")
    model = UNet2D(
        in_channels=len(channel_names),
        out_channels=len(horizons),
        depth=int(model_cfg.get("depth", 4)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)

    train_loader = DataLoader(TensorCaseDataset(train_samples), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(TensorCaseDataset(eval_samples), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_state: dict[str, torch.Tensor] | None = None
    best_eval_brier = float("inf")
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = _weighted_loss(
                logits,
                y,
                pos_weight=pos_weight,
                horizon_weights=horizon_weights.to(device),
                bce_weight=bce_weight,
                dice_weight=dice_weight,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        scheduler.step()
        eval_brier, eval_brier_per_h = _evaluate_brier(model, eval_loader, device)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "eval_brier": eval_brier,
                "eval_brier_per_horizon": eval_brier_per_h,
            }
        )
        LOGGER.info(
            "epoch=%s train_loss=%.4f eval_brier=%.6f",
            epoch,
            train_loss,
            eval_brier,
        )

        if eval_brier < best_eval_brier:
            best_eval_brier = eval_brier
            best_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                LOGGER.info("Early stopping at epoch=%s (patience=%s).", epoch, patience)
                break

    if best_state is None:
        raise RuntimeError("Training failed: no best state captured.")
    model.load_state_dict(best_state)
    model.eval()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    git_sha = _maybe_git_sha() or "unknown"
    run_id = f"{timestamp}_{git_sha}"
    run_dir = Path(config.get("model_output_root", "models/spread_v2")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run_dir / "model.pt")

    # Export ONNX (N,C,H,W -> N,T,H,W)
    dummy = torch.zeros(
        (1, len(channel_names), train_samples[0].x.shape[1], train_samples[0].x.shape[2]),
        dtype=torch.float32,
    )
    onnx_opset = int(export_cfg.get("onnx_opset", 17))
    onnx_path = run_dir / "model.onnx"
    export_kwargs = {
        "input_names": ["x"],
        "output_names": ["logits"],
        "opset_version": onnx_opset,
        "dynamic_axes": {
            "x": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
    }
    try:
        # Prefer legacy exporter path to avoid requiring onnxscript in lean envs.
        torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs)

    quantized_path = run_dir / "model.int8.onnx"
    if bool(export_cfg.get("quantize_int8", True)):
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(
                str(onnx_path),
                str(quantized_path),
                weight_type=QuantType.QInt8,
            )
        except Exception:
            LOGGER.exception("Failed to export quantized ONNX model.")

    metrics = {
        "best_epoch": int(best_epoch),
        "best_eval_brier": float(best_eval_brier),
        "history": history,
        "n_train": len(train_samples),
        "n_eval": len(eval_samples),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    feature_schema = {
        "channels": list(channel_names),
        "horizons_hours": horizons,
    }
    (run_dir / "feature_schema.json").write_text(
        json.dumps(feature_schema, indent=2) + "\n",
        encoding="utf-8",
    )
    write_contract(
        run_dir / "runtime_contract.json",
        SpreadRuntimeContract(
            channels=tuple(channel_names),
            channel_metadata=CANONICAL_CHANNEL_METADATA,
        ),
    )
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "model_name": "LearnedSpreadModelV2",
        "model_version": "v2",
        "horizons_hours": horizons,
        "channel_count": len(channel_names),
        "package_versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cpu_only": True,
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Training v2 complete: %s", run_dir)
    return run_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Train learned spread model v2.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    config = load_config(args.config)
    train_spread_v2(config)


if __name__ == "__main__":
    main()
