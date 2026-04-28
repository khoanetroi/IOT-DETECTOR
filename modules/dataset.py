"""
modules/dataset.py
IoT Flow Dataset: Load CSV -> Clean -> Normalize -> Windowing -> Augmentation -> PyTorch Dataset

References:
  - AOC-IDS (Xinchen Zhang et al., INFOCOM 2024): Contrastive augmentation strategy
  - bandwidth-estimation (Mirko Schiavone, Surrey 2025): Client-session windowing approach
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset


# ────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING CONSTANTS
# ────────────────────────────────────────────────────────────────
# Columns present in the UNSW IoT Dataset (flows.zip) CSV files
BEHAVIORAL_FEATURES = [
    "srcNumPackets",
    "dstNumPackets",
    "srcPayloadSize",
    "dstPayloadSize",
    "srcAvgPayloadSize",
    "dstAvgPayloadSize",
    "srcMaxPayloadSize",
    "dstMaxPayloadSize",
    "srcStdDevPayloadSize",
    "dstStdDevPayloadSize",
    "flowDuration",
    "srcAvgInterarrivalTime",
    "dstAvgInterarrivalTime",
    "avgInterarrivalTime",
    "srcStdDevInterarrivalTime",
    "dstStdDevInterarrivalTime",
    "stdDevInterarrivalTime",
]


# ────────────────────────────────────────────────────────────────
#  DATA LOADING
# ────────────────────────────────────────────────────────────────
def load_flow_data(data_dir: str, max_files: int = 0) -> pd.DataFrame:
    """
    Load CSV files from *data_dir*, assign a ``device_id`` derived from
    the filename, and concatenate into a single DataFrame.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing ``*_flows.csv`` files.
    max_files : int
        Maximum number of files to load (0 = load all).

    Returns
    -------
    pd.DataFrame
    """
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if max_files > 0:
        all_files = all_files[:max_files]

    frames = []
    for fpath in all_files:
        basename = os.path.basename(fpath)
        device_name = basename.split("_")[0]
        df = pd.read_csv(fpath)
        df["device_id"] = device_name
        frames.append(df)
        print(f"  [+] {device_name}: {len(df):,} rows")

    return pd.concat(frames, ignore_index=True)


# ────────────────────────────────────────────────────────────────
#  CLEANING & NORMALISATION
# ────────────────────────────────────────────────────────────────
def clean_and_normalise(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], StandardScaler]:
    """
    Select behavioural features, fill NaNs with 0, and apply
    StandardScaler normalisation.

    Returns the cleaned DataFrame, the list of selected feature
    column names, and the fitted scaler (so it can be reused
    during inference / fine-tuning).
    """
    if feature_cols is None:
        feature_cols = [c for c in BEHAVIORAL_FEATURES if c in df.columns]

    if not feature_cols:
        # Fallback: grab all numeric columns except device_id
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c != "device_id"]

    df[feature_cols] = df[feature_cols].fillna(0).astype(np.float32)

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, feature_cols, scaler


# ────────────────────────────────────────────────────────────────
#  WINDOWING (Time-series segmentation)
# ────────────────────────────────────────────────────────────────
def create_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice each device's flow history into fixed-length windows.

    Returns
    -------
    windows : np.ndarray, shape [N, window_size, num_features]
    labels  : np.ndarray, shape [N]  (integer-encoded device_id)
        Labels are kept for optional downstream fine-tuning /
        evaluation but are **not** used during pre-training.
    """
    # Encode device names to integer labels
    device_names = df["device_id"].unique().tolist()
    device_to_idx = {name: idx for idx, name in enumerate(device_names)}

    all_windows = []
    all_labels = []

    for device_name, group in df.groupby("device_id"):
        data = group[feature_cols].values.astype(np.float32)
        n_windows = len(data) // window_size

        if n_windows == 0:
            continue

        data = data[: n_windows * window_size]
        wins = data.reshape(n_windows, window_size, len(feature_cols))
        all_windows.append(wins)
        all_labels.append(
            np.full(n_windows, device_to_idx[device_name], dtype=np.int64)
        )

    windows = np.concatenate(all_windows, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return windows, labels


# ────────────────────────────────────────────────────────────────
#  CONTRASTIVE AUGMENTATIONS
# ────────────────────────────────────────────────────────────────
def random_mask(sequence: np.ndarray, mask_prob: float = 0.15) -> np.ndarray:
    """Apply random masking: set ``mask_prob`` fraction of values to 0."""
    aug = sequence.copy()
    mask = np.random.rand(*aug.shape) < mask_prob
    aug[mask] = 0.0
    return aug


def generate_positive_pair(
    sequence: np.ndarray, mask_prob: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create two independently masked views of the same sequence
    (SimCLR / MoCo-style positive pair).
    """
    return random_mask(sequence, mask_prob), random_mask(sequence, mask_prob)


# ────────────────────────────────────────────────────────────────
#  PYTORCH DATASET
# ────────────────────────────────────────────────────────────────
class FlowContrastiveDataset(Dataset):
    """
    PyTorch Dataset that yields (view_a, view_b) positive pairs
    for contrastive pre-training.

    Each call to ``__getitem__`` generates *fresh* random masks,
    so the model effectively sees different augmentations every epoch
    (infinite augmentation, like AOC-IDS).
    """

    def __init__(self, windows: np.ndarray, mask_prob: float = 0.15):
        self.windows = windows
        self.mask_prob = mask_prob

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.windows[idx]
        xa, xb = generate_positive_pair(seq, self.mask_prob)
        return (
            torch.from_numpy(xa).float(),
            torch.from_numpy(xb).float(),
        )


class FlowClassificationDataset(Dataset):
    """
    PyTorch Dataset for downstream fine-tuning (supervised).
    Returns (sequence, label).
    """

    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        self.windows = windows
        self.labels = labels

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.windows[idx]).float(),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
