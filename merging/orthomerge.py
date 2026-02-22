"""
OrthoMerge: Merge specialist models on Riemannian manifolds via Lie algebra.

Key insight: Causeway's rotation matrix R lives on O(n). OrthoMerge operates
in the Lie algebra of O(n). They share the same manifold — OrthoMerge is the
only mathematically correct way to merge these models.

Workflow:
1. Compute task vectors: Δ_i = W_i - W_init
2. Procrustes decompose each Δ: Q_i (orthogonal) + R_i (residual)
3. Merge Q's in Lie algebra: Q_merged = exp(mean(log(Q_i)))
4. Average residuals: R_merged = mean(R_i)
5. W_merged = W_init + Q_merged + R_merged
"""

import torch
import torch.nn as nn
from scipy.linalg import logm, expm
import numpy as np
from typing import List, Dict
from copy import deepcopy


def procrustes_decompose(delta: torch.Tensor) -> tuple:
    """
    SVD-based decomposition into orthogonal Q and residual R.
    Δ = U Σ V^T → Q = UV^T (nearest orthogonal), R = Δ - Q

    Args:
        delta: [m, n] task vector (weight difference)
    Returns:
        Q: [m, n] orthogonal component
        R: [m, n] residual
    """
    U, S, Vt = torch.linalg.svd(delta, full_matrices=False)
    Q = U @ Vt
    R = delta - Q
    return Q, R


def merge_orthogonal(Qs: List[torch.Tensor]) -> torch.Tensor:
    """
    Merge orthogonal matrices in Lie algebra of O(n).
    Q_merged = expm(mean(logm(Q_i)))

    For non-square matrices, pad to square, merge, then crop.
    """
    # Convert to square if needed
    m, n = Qs[0].shape
    is_square = (m == n)
    max_dim = max(m, n)

    logs = []
    for Q in Qs:
        Q_np = Q.detach().cpu().numpy().astype(np.float64)

        if not is_square:
            # Pad to square with identity
            Q_sq = np.eye(max_dim, dtype=np.float64)
            Q_sq[:m, :n] = Q_np
        else:
            Q_sq = Q_np

        # logm in Lie algebra
        log_Q = logm(Q_sq)
        # Take real part (numerical noise can add small imaginary)
        log_Q = log_Q.real
        logs.append(log_Q)

    # Average in Lie algebra
    mean_log = np.mean(logs, axis=0)

    # Map back to manifold
    Q_merged_sq = expm(mean_log).real

    # Crop back if padded
    Q_merged_np = Q_merged_sq[:m, :n]

    return torch.tensor(Q_merged_np, dtype=Qs[0].dtype, device=Qs[0].device)


def compute_task_vectors(
    init_state: Dict[str, torch.Tensor],
    trained_states: List[Dict[str, torch.Tensor]],
) -> List[Dict[str, torch.Tensor]]:
    """Compute Δ_i = W_i - W_init for each trained model."""
    task_vectors = []
    for trained in trained_states:
        delta = {}
        for key in init_state:
            if key in trained:
                delta[key] = trained[key].float() - init_state[key].float()
        task_vectors.append(delta)
    return task_vectors


def merge_models(
    init_checkpoint: str,
    specialist_checkpoints: List[str],
    output_path: str,
    merge_embeddings: bool = False,
):
    """
    Full OrthoMerge pipeline.

    Args:
        init_checkpoint: path to initial (pre-training) checkpoint
        specialist_checkpoints: list of paths to trained specialist checkpoints
        output_path: where to save merged model
        merge_embeddings: whether to merge embedding layers (usually False)
    """
    print(f"Loading init checkpoint: {init_checkpoint}")
    init_state = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" in init_state:
        init_state = init_state["model_state_dict"]

    trained_states = []
    for cp in specialist_checkpoints:
        print(f"Loading specialist: {cp}")
        state = torch.load(cp, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        trained_states.append(state)

    # Compute task vectors
    task_vectors = compute_task_vectors(init_state, trained_states)

    # Merge each parameter
    merged_state = {}
    skip_prefixes = [] if merge_embeddings else ["tok_emb", "puzzle_emb"]

    for key in init_state:
        # Skip non-float parameters
        if not init_state[key].is_floating_point():
            merged_state[key] = init_state[key]
            continue

        # Skip embeddings if requested
        if any(key.startswith(p) for p in skip_prefixes):
            merged_state[key] = init_state[key]
            continue

        deltas = [tv[key] for tv in task_vectors if key in tv]
        if not deltas:
            merged_state[key] = init_state[key]
            continue

        param = init_state[key].float()

        if param.ndim >= 2 and min(param.shape) > 1:
            # Matrix: Procrustes decompose + Lie algebra merge
            Qs = []
            Rs = []
            for delta in deltas:
                Q, R = procrustes_decompose(delta)
                Qs.append(Q)
                Rs.append(R)

            Q_merged = merge_orthogonal(Qs)
            R_merged = torch.mean(torch.stack(Rs), dim=0)
            merged_state[key] = (param + Q_merged + R_merged).to(init_state[key].dtype)
        else:
            # Scalar/vector: simple average of task vectors
            avg_delta = torch.mean(torch.stack(deltas), dim=0)
            merged_state[key] = (param + avg_delta).to(init_state[key].dtype)

        print(f"  Merged: {key} {list(param.shape)}")

    torch.save({"model_state_dict": merged_state}, output_path)
    print(f"Merged model saved to {output_path}")
    return merged_state
