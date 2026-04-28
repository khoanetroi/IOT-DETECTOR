"""
run_pretrain.py
Entry-point for Self-Supervised Pre-training of IoT Flow Transformer.

Usage:
    python run_pretrain.py                              # default config
    python run_pretrain.py --config configs/custom.json # custom config

This script:
  1. Loads and preprocesses real UNSW IoT flow CSVs
  2. Creates contrastive (SimCLR-style) positive pairs via random masking
  3. Trains a Transformer Encoder with NT-Xent loss
  4. Saves checkpoints to experiments/ for downstream fine-tuning
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# Ensure project root is on path (for embedded python)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.dataset import (
    load_flow_data,
    clean_and_normalise,
    create_windows,
    FlowContrastiveDataset,
)
from modules.models import FlowTransformerEncoder
from modules.losses import NTXentLoss


# ────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "pretrain.json"
)


# ────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ────────────────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches = len(dataloader)

    for batch_idx, (xa, xb) in enumerate(dataloader):
        xa, xb = xa.to(device), xb.to(device)

        # Forward
        z_a = model(xa)
        z_b = model(xb)

        loss = criterion(z_a, z_b)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 20 == 0:
            print(
                f"  [Epoch {epoch}] Batch {batch_idx:>4d}/{n_batches} "
                f"| Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / n_batches
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    for xa, xb in dataloader:
        xa, xb = xa.to(device), xb.to(device)
        z_a = model(xa)
        z_b = model(xb)
        total_loss += criterion(z_a, z_b).item()
    return total_loss / len(dataloader)


# ────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pre-train IoT Flow Transformer")
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG, help="Path to JSON config"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  IoT Flow Transformer - Contrastive Pre-training")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── 1. Load data ──────────────────────────────────────────
    print("[Phase 1] Loading flow data...")
    df = load_flow_data(data_cfg["data_dir"], max_files=data_cfg["max_files"])
    print(f"  Total rows: {len(df):,}\n")

    # ── 2. Clean & normalise ──────────────────────────────────
    print("[Phase 1] Cleaning & normalising features...")
    df, feature_cols, scaler = clean_and_normalise(df)
    print(f"  Selected {len(feature_cols)} features: {feature_cols}\n")

    # ── 3. Windowing ──────────────────────────────────────────
    print("[Phase 1] Creating windows...")
    windows, labels = create_windows(df, feature_cols, window_size=data_cfg["window_size"])
    print(f"  Windows shape: {windows.shape}  (samples, time-steps, features)")
    print(f"  Unique devices: {len(np.unique(labels))}\n")

    # ── 4. Train / Val split ──────────────────────────────────
    dataset = FlowContrastiveDataset(windows, mask_prob=data_cfg["mask_prob"])
    val_size = int(len(dataset) * data_cfg["val_ratio"])
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False, drop_last=False
    )
    print(f"  Train: {train_size:,} samples | Val: {val_size:,} samples\n")

    # ── 5. Build model ────────────────────────────────────────
    num_features = windows.shape[2]
    model = FlowTransformerEncoder(
        input_dim=num_features,
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        proj_dim=model_cfg["proj_dim"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Phase 2] Model initialised ({total_params:,} parameters)\n")

    criterion = NTXentLoss(temperature=train_cfg["temperature"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["num_epochs"]
    )

    # ── 6. Training loop ──────────────────────────────────────
    ckpt_dir = train_cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")

    print(f"[Phase 2] Starting pre-training for {train_cfg['num_epochs']} epochs...")
    print(f"{'─'*60}")

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"{'─'*60}\n"
            f"  Epoch {epoch}/{train_cfg['num_epochs']} "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| LR: {lr_now:.6f} "
            f"| Time: {elapsed:.1f}s"
        )

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"pretrain_epoch_{epoch}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "config": cfg,
                "feature_cols": feature_cols,
            },
            ckpt_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(ckpt_dir, "pretrain_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "config": cfg,
                    "feature_cols": feature_cols,
                },
                best_path,
            )
            print(f"  ** New best model saved -> {best_path}")

        print(f"  Checkpoint saved -> {ckpt_path}")
        print(f"{'─'*60}\n")

    print("=" * 60)
    print("  PRE-TRAINING COMPLETE!")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved in: {os.path.abspath(ckpt_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
