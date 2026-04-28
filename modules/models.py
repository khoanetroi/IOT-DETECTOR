"""
modules/models.py
Transformer Encoder for Self-Supervised IoT Device Fingerprinting.

Architecture inspired by:
  - bandwidth-estimation (Schiavone 2025): Transformer encoder w/ positional encoding
  - AOC-IDS (Zhang et al. 2024): Detachable projection head for contrastive pre-training

The model is split into two logical parts:
  1. TransformerEncoder  - learns rich representations from flow windows
  2. ProjectionHead       - maps representations to a compact space for
                            contrastive loss (removed during fine-tuning)
"""

import math
import torch
import torch.nn as nn


# ────────────────────────────────────────────────────────────────
#  POSITIONAL ENCODING (Standard sinusoidal, Vaswani et al. 2017)
# ────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """Injects positional information into the sequence."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ────────────────────────────────────────────────────────────────
#  PROJECTION HEAD (Contrastive Learning)
# ────────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """
    Two-layer MLP that projects encoder representations into a
    lower-dimensional space where contrastive loss is computed.
    Detached / replaced during downstream fine-tuning.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ────────────────────────────────────────────────────────────────
#  TRANSFORMER ENCODER (Core "Brain")
# ────────────────────────────────────────────────────────────────
class FlowTransformerEncoder(nn.Module):
    """
    Transformer-based encoder that converts a window of IoT flow
    features into a fixed-length representation vector.

    Parameters
    ----------
    input_dim : int
        Number of input features per time-step.
    d_model : int
        Internal dimensionality of the transformer.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of TransformerEncoderLayer blocks.
    dim_feedforward : int
        Hidden size of the feed-forward sub-layer.
    dropout : float
        Dropout probability.
    proj_dim : int
        Output dimensionality of the contrastive projection head.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        proj_dim: int = 64,
    ):
        super().__init__()

        # --- Linear projection to d_model ---
        self.input_projection = nn.Linear(input_dim, d_model)

        # --- Positional Encoding ---
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # --- Transformer Encoder Stack ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",  # smoother activation than ReLU
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # --- Layer norm on the output (stabilises training) ---
        self.norm = nn.LayerNorm(d_model)

        # --- Projection Head (contrastive) ---
        self.projection_head = ProjectionHead(
            input_dim=d_model, output_dim=proj_dim
        )

        self.d_model = d_model

    # ----- helpers --------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the representation vector **without** the projection head.
        Use this during fine-tuning or feature extraction.
        """
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.norm(x)
        # Use [CLS]-like strategy: take the mean across time steps
        return x.mean(dim=1)  # [batch, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass including projection head.
        Used during contrastive pre-training.
        """
        h = self.encode(x)            # [batch, d_model]
        z = self.projection_head(h)   # [batch, proj_dim]
        return z


# ────────────────────────────────────────────────────────────────
#  CLASSIFIER HEAD (Fine-tuning)
# ────────────────────────────────────────────────────────────────
class FlowClassifier(nn.Module):
    """
    Wraps a pre-trained FlowTransformerEncoder with a classification
    head for supervised device fingerprinting (Phase 3).
    """

    def __init__(self, encoder: FlowTransformerEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.encoder.encode(x)  # frozen encoder
        return self.classifier(h)
