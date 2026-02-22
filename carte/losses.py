"""
CARTELoss: Combined loss function.

L = L_lm (stablemax CE) + 0.5 * L_halt (BCE) + λ_acyclic * L_acyclic + λ_sparse * L_sparse + λ_ortho * L_ortho

Acyclicity loss ramped from 0 → λ over first 30% of training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def stablemax_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Stablemax cross-entropy: uses log-sum-exp trick for numerical stability.
    More stable than standard softmax CE for low-precision training.

    Args:
        logits: [B, S, V] raw logits
        targets: [B, S] target indices
    Returns:
        scalar loss
    """
    B, S, V = logits.shape
    logits_flat = logits.view(-1, V)
    targets_flat = targets.view(-1)

    # Log-softmax with max subtraction for stability
    max_logits = logits_flat.max(dim=-1, keepdim=True).values
    shifted = logits_flat - max_logits
    log_sum_exp = shifted.exp().sum(dim=-1).log()
    target_logits = shifted.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
    loss = log_sum_exp - target_logits

    # Mask padding (target == -100)
    mask = targets_flat != -100
    if mask.any():
        return loss[mask].mean()
    return loss.mean()


class CARTELoss(nn.Module):
    """Combined CARTE loss with acyclicity ramp."""

    def __init__(
        self,
        lambda_acyclic: float = 1.0,
        lambda_sparse: float = 0.01,
        lambda_ortho: float = 0.1,
        halt_weight: float = 0.5,
        ramp_fraction: float = 0.3,
    ):
        super().__init__()
        self.lambda_acyclic = lambda_acyclic
        self.lambda_sparse = lambda_sparse
        self.lambda_ortho = lambda_ortho
        self.halt_weight = halt_weight
        self.ramp_fraction = ramp_fraction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        halt_probs: torch.Tensor,
        halt_targets: torch.Tensor,
        causal_losses: dict,
        step: int,
        total_steps: int,
    ) -> tuple:
        """
        Args:
            logits: [B, S, V]
            targets: [B, S]
            halt_probs: [B, S] predicted halt probability
            halt_targets: [B, S] binary halt targets (1 = should output here)
            causal_losses: dict with 'acyclic', 'sparse', 'ortho' tensors
            step: current training step
            total_steps: total training steps
        Returns:
            (total_loss, loss_dict)
        """
        # Language modeling loss
        L_lm = stablemax_ce(logits, targets)

        # Halt loss
        L_halt = F.binary_cross_entropy(halt_probs, halt_targets.float())

        # Causal regularization with acyclicity ramp
        progress = step / max(total_steps, 1)
        acyclic_scale = min(progress / self.ramp_fraction, 1.0) if self.ramp_fraction > 0 else 1.0

        L_acyclic = causal_losses["acyclic"]
        L_sparse = causal_losses["sparse"]
        L_ortho = causal_losses["ortho"]

        total = (
            L_lm
            + self.halt_weight * L_halt
            + self.lambda_acyclic * acyclic_scale * L_acyclic
            + self.lambda_sparse * L_sparse
            + self.lambda_ortho * L_ortho
        )

        loss_dict = {
            "total": total.item(),
            "lm": L_lm.item(),
            "halt": L_halt.item(),
            "acyclic": L_acyclic.item(),
            "sparse": L_sparse.item(),
            "ortho": L_ortho.item(),
            "acyclic_scale": round(acyclic_scale, 3),
        }

        return total, loss_dict
