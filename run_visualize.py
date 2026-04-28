"""
run_visualize.py
Generate t-SNE visualization of Transformer embeddings + Confusion Matrix.

This script loads the fine-tuned model, extracts embeddings from all devices,
and produces publication-quality plots proving the encoder learns separable
behavioral clusters.

Usage:
    python run_visualize.py
"""

import os
import sys
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.dataset import (
    load_flow_data,
    clean_and_normalise,
    create_windows,
)
from modules.models import FlowTransformerEncoder


# ────────────────────────────────────────────────────────────────
def extract_embeddings(encoder, windows, device, batch_size=256):
    """Run all windows through the encoder and collect embeddings."""
    encoder.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[i : i + batch_size]).float().to(device)
            emb = encoder.encode(batch)  # [batch, d_model]
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


# ────────────────────────────────────────────────────────────────
def plot_tsne(embeddings, labels, idx_to_device, save_path, title="t-SNE IoT Device Embeddings"):
    """Generate a beautiful t-SNE scatter plot."""
    print("  Running t-SNE (this may take a minute)...")
    
    # Subsample if too many points (t-SNE is O(n^2))
    max_points = 8000
    if len(embeddings) > max_points:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    # Beautiful color palette
    unique_labels = sorted(np.unique(labels))
    n_classes = len(unique_labels)
    cmap = plt.cm.get_cmap("tab20", n_classes)

    fig, ax = plt.subplots(figsize=(14, 10))
    for i, label_id in enumerate(unique_labels):
        mask = labels == label_id
        name = idx_to_device.get(label_id, f"Device {label_id}")
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(i)], label=name, alpha=0.6, s=15, edgecolors="none"
        )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, markerscale=2, framealpha=0.9
    )
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  t-SNE plot saved -> {save_path}")


# ────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, idx_to_device, save_path):
    """Generate and save a confusion matrix plot."""
    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    target_names = [idx_to_device.get(i, f"D{i}") for i in unique_labels]

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    fig, ax = plt.subplots(figsize=(max(10, len(unique_labels)), max(8, len(unique_labels) * 0.8)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    ax.set_title("Confusion Matrix - IoT Device Classification", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved -> {save_path}")


# ────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "experiments"
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load fine-tuned checkpoint ─────────────────────────
    ckpt_path = os.path.join(output_dir, "finetune_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print("  Please run run_finetune.py first.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  IoT Flow Transformer - Visualization & Analysis")
    print(f"{'='*60}\n")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    idx_to_device = checkpoint["idx_to_device"]
    feature_cols = checkpoint["feature_cols"]
    pretrain_cfg = checkpoint["pretrain_config"]["model"]

    # ── 2. Rebuild encoder ────────────────────────────────────
    print("[Step 1] Rebuilding encoder from checkpoint...")
    encoder = FlowTransformerEncoder(
        input_dim=len(feature_cols),
        d_model=pretrain_cfg["d_model"],
        nhead=pretrain_cfg["nhead"],
        num_layers=pretrain_cfg["num_layers"],
        dim_feedforward=pretrain_cfg["dim_feedforward"],
        dropout=pretrain_cfg["dropout"],
        proj_dim=pretrain_cfg["proj_dim"],
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    print("  Encoder loaded.\n")

    # ── 3. Load data ──────────────────────────────────────────
    ft_cfg = checkpoint["config"]
    data_cfg = ft_cfg["data"]

    print("[Step 2] Loading & preprocessing data...")
    df = load_flow_data(data_cfg["data_dir"], max_files=data_cfg["max_files"])
    df, _, _ = clean_and_normalise(df, feature_cols=feature_cols)
    windows, labels = create_windows(df, feature_cols, window_size=data_cfg["window_size"])
    print(f"  Total windows: {windows.shape[0]:,}\n")

    # ── 4. Extract embeddings ─────────────────────────────────
    print("[Step 3] Extracting embeddings from Transformer...")
    embeddings = extract_embeddings(encoder, windows, device)
    print(f"  Embedding shape: {embeddings.shape}\n")

    # ── 5. t-SNE ──────────────────────────────────────────────
    print("[Step 4] Generating t-SNE visualization...")

    # a) Pre-trained embeddings t-SNE
    tsne_path = os.path.join(output_dir, "tsne_embeddings.png")
    plot_tsne(embeddings, labels, idx_to_device, tsne_path,
              title="t-SNE: IoT Device Behavioral Embeddings (Fine-tuned Transformer)")

    # ── 6. Confusion matrix (from saved predictions) ──────────
    print("\n[Step 5] Generating confusion matrix...")
    
    # Quick inference pass for confusion matrix
    from modules.models import FlowClassifier
    classifier = FlowClassifier(encoder, num_classes=checkpoint["num_classes"]).to(device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(windows), 256):
            batch = torch.from_numpy(windows[i:i+256]).float().to(device)
            batch_labels = labels[i:i+256]
            logits = classifier(batch)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(all_labels, all_preds, idx_to_device, cm_path)

    # ── 7. Summary ────────────────────────────────────────────
    overall_acc = (all_preds == all_labels).mean()
    print(f"\n{'='*60}")
    print(f"  VISUALIZATION COMPLETE!")
    print(f"  Overall Accuracy: {overall_acc:.4f}")
    print(f"  Files generated:")
    print(f"    - {tsne_path}")
    print(f"    - {cm_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
