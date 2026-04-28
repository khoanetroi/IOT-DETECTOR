"""
run_finetune.py
Fine-tune the pre-trained Transformer Encoder for IoT Device Classification.

Strategy (inspired by AOC-IDS & bandwidth-estimation):
  Phase A: Freeze encoder, train only classifier head (few epochs)
  Phase B: Unfreeze encoder, fine-tune everything with lower LR

Usage:
    python run_finetune.py
    python run_finetune.py --config configs/finetune.json
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.dataset import (
    load_flow_data,
    clean_and_normalise,
    create_windows,
    FlowClassificationDataset,
)
from modules.models import FlowTransformerEncoder, FlowClassifier


# ────────────────────────────────────────────────────────────────
def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "finetune.json"
)


# ────────────────────────────────────────────────────────────────
def split_indices(n, val_ratio, test_ratio, seed=42):
    """Stratification-free random split into train/val/test indices."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    return train_idx, val_idx, test_idx


# ────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fine-tune IoT Device Classifier")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  IoT Device Classifier - Fine-tuning")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── 1. Load & preprocess ──────────────────────────────────
    print("[Step 1] Loading flow data...")
    df = load_flow_data(data_cfg["data_dir"], max_files=data_cfg["max_files"])
    print(f"  Total: {len(df):,} rows\n")

    print("[Step 2] Cleaning & normalising...")
    df, feature_cols, scaler = clean_and_normalise(df)
    print(f"  Features: {len(feature_cols)}\n")

    print("[Step 3] Creating windows + labels...")
    windows, labels = create_windows(df, feature_cols, window_size=data_cfg["window_size"])
    num_classes = len(np.unique(labels))
    print(f"  Windows: {windows.shape[0]:,} | Classes: {num_classes}\n")

    # Build device name mapping
    device_names = df["device_id"].unique().tolist()
    idx_to_device = {i: name for i, name in enumerate(device_names)}

    # ── 2. Split ──────────────────────────────────────────────
    train_idx, val_idx, test_idx = split_indices(
        len(windows), data_cfg["val_ratio"], data_cfg["test_ratio"]
    )
    full_dataset = FlowClassificationDataset(windows, labels)
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=train_cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=train_cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=train_cfg["batch_size"], shuffle=False)
    print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}\n")

    # ── 3. Load pre-trained encoder ──────────────────────────
    ckpt_path = model_cfg["pretrain_checkpoint"]
    print(f"[Step 4] Loading pre-trained checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    pretrain_cfg = checkpoint["config"]["model"]

    encoder = FlowTransformerEncoder(
        input_dim=windows.shape[2],
        d_model=pretrain_cfg["d_model"],
        nhead=pretrain_cfg["nhead"],
        num_layers=pretrain_cfg["num_layers"],
        dim_feedforward=pretrain_cfg["dim_feedforward"],
        dropout=pretrain_cfg["dropout"],
        proj_dim=pretrain_cfg["proj_dim"],
    )
    encoder.load_state_dict(checkpoint["model_state_dict"])
    print("  Encoder weights loaded successfully!\n")

    # ── 4. Build classifier ───────────────────────────────────
    model = FlowClassifier(encoder, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    freeze_epochs = model_cfg["freeze_encoder_epochs"]
    total_epochs = train_cfg["num_epochs"]
    ckpt_dir = train_cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_acc = 0.0

    # ── 5. Training ───────────────────────────────────────────
    print(f"[Step 5] Training for {total_epochs} epochs")
    print(f"  Phase A (epochs 1-{freeze_epochs}): Encoder FROZEN, train classifier only")
    print(f"  Phase B (epochs {freeze_epochs+1}-{total_epochs}): Encoder UNFROZEN, full fine-tune")
    print(f"{'─'*60}\n")

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()

        # --- Phase A: Freeze encoder ---
        if epoch <= freeze_epochs:
            for p in model.encoder.parameters():
                p.requires_grad = False
            trainable = model.classifier.parameters()
            lr = train_cfg["learning_rate"]
        else:
            # --- Phase B: Unfreeze encoder with lower LR ---
            for p in model.encoder.parameters():
                p.requires_grad = True
            trainable = [
                {"params": model.encoder.parameters(), "lr": train_cfg["learning_rate"] * model_cfg["unfreeze_lr_factor"]},
                {"params": model.classifier.parameters()},
            ]
            lr = train_cfg["learning_rate"]

        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=train_cfg["weight_decay"])

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        phase = "A (frozen)" if epoch <= freeze_epochs else "B (unfrozen)"
        print(
            f"  Epoch {epoch:>2d}/{total_epochs} [{phase}] "
            f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
            f"| {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(ckpt_dir, "finetune_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes,
                "idx_to_device": idx_to_device,
                "feature_cols": feature_cols,
                "config": cfg,
                "pretrain_config": checkpoint["config"],
            }, best_path)
            print(f"         ** Best model saved -> {best_path} (Acc: {val_acc:.4f})")

    # ── 6. Test evaluation ────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL TEST EVALUATION")
    print(f"{'='*60}\n")

    # Reload best model
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss:     {test_loss:.4f}\n")

    target_names = [idx_to_device[i] for i in sorted(idx_to_device.keys())]
    report = classification_report(test_labels, test_preds, target_names=target_names, zero_division=0)
    print("  Classification Report:")
    print(report)

    # Save report
    report_path = os.path.join(ckpt_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(report)
    print(f"  Report saved -> {report_path}")

    print(f"\n{'='*60}")
    print(f"  FINE-TUNING COMPLETE!")
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Test Accuracy:     {test_acc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
