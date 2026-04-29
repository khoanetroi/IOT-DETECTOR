"""
modules/inference.py
Production-ready inference module for IoT Device Fingerprinting.

This is the "one-call" module you plug into any API / Cloud service.
It handles everything: load model, preprocess raw flows, predict device.

Usage (from Python):
    from modules.inference import IoTFingerprinter
    
    fp = IoTFingerprinter("experiments/finetune_best.pth")
    result = fp.predict_from_csv("new_device_traffic.csv")
    print(result)  # {'device': 'AmazonEcho', 'confidence': 0.95, ...}

Usage (from API / Cloud):
    Just wrap IoTFingerprinter in a FastAPI/Flask route.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.dataset import BEHAVIORAL_FEATURES, create_windows
from modules.models import FlowTransformerEncoder, FlowClassifier


class IoTFingerprinter:
    """
    End-to-end IoT device identification from raw flow data.
    
    Load once, predict many times. Suitable for:
      - REST API (FastAPI / Flask)
      - Cloud Functions (AWS Lambda / Google Cloud Run)
      - Edge deployment (Raspberry Pi gateway)
    """

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.idx_to_device = ckpt["idx_to_device"]
        self.feature_cols = ckpt["feature_cols"]
        self.num_classes = ckpt["num_classes"]
        self.window_size = ckpt["config"]["data"]["window_size"]
        
        # Rebuild model
        pcfg = ckpt["pretrain_config"]["model"]
        encoder = FlowTransformerEncoder(
            input_dim=len(self.feature_cols),
            d_model=pcfg["d_model"],
            nhead=pcfg["nhead"],
            num_layers=pcfg["num_layers"],
            dim_feedforward=pcfg["dim_feedforward"],
            dropout=pcfg["dropout"],
            proj_dim=pcfg["proj_dim"],
        )
        self.model = FlowClassifier(encoder, num_classes=self.num_classes)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load scaler from checkpoint (same normalisation as training)
        self.scaler_mean = np.array(ckpt.get("scaler_mean", None))
        self.scaler_scale = np.array(ckpt.get("scaler_scale", None))
        has_scaler = self.scaler_mean is not None and self.scaler_scale is not None

        print(f"[IoTFingerprinter] Model loaded: {self.num_classes} device classes")
        print(f"  Features: {self.feature_cols}")
        print(f"  Window size: {self.window_size}")
        print(f"  Scaler: {'from training data' if has_scaler else 'per-file (fallback)'}")
        print(f"  Device: {self.device}")

    def _preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Clean and normalise a raw DataFrame, return windows."""
        cols = [c for c in self.feature_cols if c in df.columns]
        if not cols:
            raise ValueError(
                f"Input CSV must contain at least some of: {self.feature_cols}"
            )

        data = df[cols].fillna(0).astype(np.float32).values

        # Use training scaler if available, else fallback to per-file z-score
        if self.scaler_mean is not None and self.scaler_scale is not None:
            data = (data - self.scaler_mean) / (self.scaler_scale + 1e-8)
        else:
            mean = data.mean(axis=0)
            std = data.std(axis=0) + 1e-8
            data = (data - mean) / std

        # Window
        n = len(data) // self.window_size
        if n == 0:
            raise ValueError(
                f"Need at least {self.window_size} rows, got {len(data)}"
            )
        data = data[: n * self.window_size]
        windows = data.reshape(n, self.window_size, len(cols))
        return windows

    @torch.no_grad()
    def predict(self, windows: np.ndarray) -> list[dict]:
        """
        Predict device for each window.
        
        Returns list of dicts with 'device', 'confidence', 'class_id'.
        """
        tensor = torch.from_numpy(windows).float().to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

        results = []
        for pred, conf in zip(preds.cpu().numpy(), confs.cpu().numpy()):
            results.append({
                "device": self.idx_to_device[int(pred)],
                "confidence": float(conf),
                "class_id": int(pred),
            })
        return results

    def predict_from_csv(self, csv_path: str) -> dict:
        """
        Load a CSV file, predict the dominant device type.
        Uses majority voting across all windows.
        """
        df = pd.read_csv(csv_path)
        windows = self._preprocess(df)
        results = self.predict(windows)

        # Majority vote
        from collections import Counter
        votes = Counter(r["device"] for r in results)
        dominant = votes.most_common(1)[0]
        avg_conf = np.mean([r["confidence"] for r in results if r["device"] == dominant[0]])

        return {
            "predicted_device": dominant[0],
            "vote_count": dominant[1],
            "total_windows": len(results),
            "average_confidence": float(avg_conf),
            "all_votes": dict(votes),
        }

    def predict_from_dataframe(self, df: pd.DataFrame) -> dict:
        """Same as predict_from_csv but accepts a DataFrame directly."""
        windows = self._preprocess(df)
        results = self.predict(windows)

        from collections import Counter
        votes = Counter(r["device"] for r in results)
        dominant = votes.most_common(1)[0]
        avg_conf = np.mean([r["confidence"] for r in results if r["device"] == dominant[0]])

        return {
            "predicted_device": dominant[0],
            "vote_count": dominant[1],
            "total_windows": len(results),
            "average_confidence": float(avg_conf),
            "all_votes": dict(votes),
        }

    def get_embedding(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract raw embeddings (for t-SNE, anomaly detection, etc.)
        """
        self.model.eval()
        tensor = torch.from_numpy(windows).float().to(self.device)
        with torch.no_grad():
            emb = self.model.encoder.encode(tensor)
        return emb.cpu().numpy()


# ────────────────────────────────────────────────────────────────
#  Quick test
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ckpt = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments", "finetune_best.pth"
    )

    if not os.path.exists(ckpt):
        print("[ERROR] Run run_finetune.py first to generate finetune_best.pth")
        sys.exit(1)

    fp = IoTFingerprinter(ckpt)
    
    # Test with one of the training CSVs
    test_csv = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "flows"
    )
    import glob
    csvs = glob.glob(os.path.join(test_csv, "*.csv"))
    if csvs:
        print(f"\n--- Testing with: {os.path.basename(csvs[0])} ---")
        result = fp.predict_from_csv(csvs[0])
        print(f"  Predicted: {result['predicted_device']}")
        print(f"  Confidence: {result['average_confidence']:.4f}")
        print(f"  Votes: {result['all_votes']}")
