"""
modules/losses.py
NT-Xent (Normalised Temperature-scaled Cross-Entropy) loss for
SimCLR / MoCo-style contrastive pre-training.

Reference:
  - "A Simple Framework for Contrastive Learning of Visual Representations"
    (Chen et al., ICML 2020)
  - AOC-IDS (Zhang et al., INFOCOM 2024): Applied NT-Xent to network flows
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent contrastive loss.

    Given a batch of N pairs (z_i, z_j), we form 2N examples.
    For each anchor z_i, the corresponding z_j is the *positive*
    and all other 2(N-1) examples are *negatives*.

    Parameters
    ----------
    temperature : float
        Scaling factor for the cosine similarity (default 0.5).
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_i, z_j : Tensor, shape [N, D]
            L2-normalised projection vectors of the two augmented views.

        Returns
        -------
        Scalar loss.
        """
        batch_size = z_i.size(0)
        device = z_i.device

        # L2 normalise
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate: [z_0, z_1, ..., z_{N-1}, z'_0, z'_1, ..., z'_{N-1}]
        representations = torch.cat([z_i, z_j], dim=0)  # [2N, D]

        # Pairwise cosine similarity matrix [2N, 2N]
        sim_matrix = torch.mm(representations, representations.t()) / self.temperature

        # For each row i, the positive is at position (i + N) mod 2N
        # Build positive-pair indices
        pos_top = torch.arange(batch_size, 2 * batch_size, device=device)  # for z_i rows
        pos_bot = torch.arange(0, batch_size, device=device)               # for z_j rows
        positive_indices = torch.cat([pos_top, pos_bot], dim=0)            # [2N]

        # Mask out self-similarity (diagonal)
        mask = (~torch.eye(2 * batch_size, dtype=torch.bool, device=device)).float()

        # Apply mask (set diagonal to very large negative so exp -> 0)
        sim_matrix = sim_matrix * mask + (~mask.bool()).float() * (-1e9)

        # Numerator: exp(sim(z_i, z_j) / tau)
        positives = sim_matrix[torch.arange(2 * batch_size, device=device), positive_indices]

        # Denominator: sum of exp over all non-self entries per row
        denominator = torch.logsumexp(sim_matrix, dim=1)

        # Loss = -log( exp(pos) / sum(exp(all)) ) = -pos + logsumexp
        loss = (-positives + denominator).mean()

        return loss
