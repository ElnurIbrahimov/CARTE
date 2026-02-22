"""
CausalTrack: Core CARTE innovation.

Each recursion step refines a sparse causal graph alongside the TRM's z_L/z_H.

Pipeline per step:
1. Pool z_L → z_pooled [B, D] (last-token)
2. Encode: v = StateEncoder(z_pooled) → [B, d_causal]
3. Edge prediction: edge_adj = Linear(v) → [B, d_causal, d_causal] (input-dependent!)
4. Graph + message passing: v' = CausalGraph(v, edge_adj) with temp annealing
5. Decode + gated inject: z_L += sigmoid(gate) * Linear(v').unsqueeze(1)

Adapted from Causeway (C:\\Users\\asus\\Desktop\\causeway\\causeway\\):
- StateEncoder: QR rotation + slimmed MLP
- CausalGraph: Gumbel-sigmoid gating + NOTEARS acyclicity + batched message passing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CARTEStateEncoder(nn.Module):
    """
    Projects pooled hidden states into causal variable space.
    Adapted from Causeway's StateEncoder — slimmed to single-layer refinement.
    """

    def __init__(self, d_model: int, d_causal: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_causal = d_causal

        # QR-initialized rotation matrix
        raw = torch.randn(d_model, d_causal)
        Q, _ = torch.linalg.qr(raw)
        self.rotation = nn.Parameter(Q[:, :d_causal].clone())

        # Single-layer refinement (slimmed from Causeway's 3-layer)
        hidden = d_causal * 2
        self.refine = nn.Sequential(
            nn.Linear(d_causal, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_causal),
        )
        self.norm = nn.LayerNorm(d_causal)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, d_model] pooled hidden state
        Returns:
            z: [B, d_causal] causal variables
        """
        h = h.float()
        z = h @ self.rotation
        z = z + self.refine(z)
        return self.norm(z)

    def orthogonality_loss(self) -> torch.Tensor:
        """||R^T R - I||_F^2"""
        RtR = self.rotation.T @ self.rotation
        I = torch.eye(self.d_causal, device=self.rotation.device)
        return torch.norm(RtR - I, p="fro") ** 2


class CARTECausalGraph(nn.Module):
    """
    Sparse DAG with input-dependent edges and Gumbel-sigmoid gating.
    Adapted from Causeway's CausalGraphLayer — modified for batched input-dependent adjacency.
    """

    def __init__(
        self,
        d_causal: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        edge_prior: float = -2.0,
    ):
        super().__init__()
        self.d_causal = d_causal
        self.num_layers = num_layers

        # Global graph parameters (combined with input-dependent edges)
        self.W_raw = nn.Parameter(torch.randn(d_causal, d_causal) * 0.01)
        self.edge_logits = nn.Parameter(torch.full((d_causal, d_causal), edge_prior))

        # Temperature buffer (set externally per step)
        self.register_buffer("temperature", torch.tensor(1.0))

        # Message passing per layer
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_causal, d_causal * 2),
                nn.GELU(),
                nn.Linear(d_causal * 2, d_causal),
            )
            for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(d_causal)) for _ in range(num_layers)
        ])
        self.scales = nn.ParameterList([
            nn.Parameter(torch.ones(d_causal)) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_causal)
        self.register_buffer("diag_mask", 1.0 - torch.eye(d_causal))

    def _gumbel_sigmoid(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if self.training:
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            gumbel_noise = torch.log(u) - torch.log(1 - u)
            return torch.sigmoid((logits + gumbel_noise) / temperature)
        else:
            return (torch.sigmoid(logits) > 0.5).float()

    def _get_global_adjacency(self, temperature: float = 1.0) -> torch.Tensor:
        """Global gated adjacency [d_causal, d_causal]."""
        gates = self._gumbel_sigmoid(self.edge_logits, temperature) * self.diag_mask
        return gates * self.W_raw * self.diag_mask

    def forward(
        self, z: torch.Tensor, edge_adj: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Message passing with combined global + input-dependent adjacency.

        Args:
            z: [B, d_causal] causal variables
            edge_adj: [B, d_causal, d_causal] input-dependent edge weights (optional)
        Returns:
            z_out: [B, d_causal] updated variables
        """
        B = z.shape[0]
        # Global adjacency: expand to batch
        W_global = self._get_global_adjacency(temperature).unsqueeze(0).expand(B, -1, -1)

        if edge_adj is not None:
            # Combine global + input-dependent edges
            diag = self.diag_mask.unsqueeze(0)
            W = (W_global + edge_adj * diag)
        else:
            W = W_global

        h = z
        for i in range(self.num_layers):
            # Batched message passing: [B, d] @ [B, d, d] → [B, d]
            messages = torch.bmm(h.unsqueeze(1), W).squeeze(1)
            messages = self.edge_mlps[i](messages)
            messages = messages * self.scales[i] + self.biases[i]
            h = h + self.dropout(F.gelu(messages))

        return self.norm(h)

    def acyclicity_loss(self, edge_adj: Optional[torch.Tensor] = None, temperature: float = 1.0) -> torch.Tensor:
        """
        NOTEARS: h(W) = tr(e^{W∘W}) - d
        If edge_adj provided, average across batch.
        """
        d = self.d_causal

        if edge_adj is not None:
            # Batch acyclicity on combined graph
            W_global = self._get_global_adjacency(temperature).unsqueeze(0).expand(edge_adj.shape[0], -1, -1)
            diag = self.diag_mask.unsqueeze(0)
            W = (W_global + edge_adj * diag)
            W_sq = W * W
            I = torch.eye(d, device=W.device).unsqueeze(0).expand(W.shape[0], -1, -1)
            M = I.clone()
            term = I.clone()
            for k in range(1, d + 1):
                term = torch.bmm(term, W_sq) / k
                M = M + term
            return (M.diagonal(dim1=-2, dim2=-1).sum(-1) - d).mean()
        else:
            W = self._get_global_adjacency(temperature)
            W_sq = W * W
            I = torch.eye(d, device=W.device)
            M = I.clone()
            term = I.clone()
            for k in range(1, d + 1):
                term = term @ W_sq / k
                M = M + term
            return torch.trace(M) - d

    def sparsity_loss(self, edge_adj: Optional[torch.Tensor] = None, temperature: float = 1.0) -> torch.Tensor:
        """L1 on adjacency."""
        loss = torch.sum(torch.abs(self._get_global_adjacency(temperature)))
        if edge_adj is not None:
            loss = loss + edge_adj.abs().mean()
        return loss

    def get_graph_stats(self) -> dict:
        with torch.no_grad():
            probs = torch.sigmoid(self.edge_logits) * self.diag_mask
            hard_edges = (probs > 0.5).sum().item()
            return {
                "hard_edges": int(hard_edges),
                "expected_edges": f"{probs.sum().item():.1f}",
                "density": round(hard_edges / (self.d_causal * (self.d_causal - 1)), 4),
                "max_edge_prob": round(probs.max().item(), 4),
                "temperature": round(self.temperature.item(), 4),
            }

    def get_top_edges(self, top_k: int = 20) -> list:
        with torch.no_grad():
            probs = torch.sigmoid(self.edge_logits) * self.diag_mask
            weights = (self.W_raw * self.diag_mask).abs()
            edges = []
            for i in range(self.d_causal):
                for j in range(self.d_causal):
                    if i != j and probs[i, j] > 0.1:
                        edges.append((i, j, round(probs[i, j].item(), 4), round(weights[i, j].item(), 6)))
            edges.sort(key=lambda x: -x[2])
            return edges[:top_k]


class CausalTrack(nn.Module):
    """
    Orchestrator: StateEncoder → edge_predictor → CausalGraph → decoder → gated injection.

    Called between z_L and z_H updates in the TRM recursive loop.
    Temperature anneals from 1.0 → 0.1 across recursion steps.
    """

    def __init__(
        self,
        d_model: int,
        d_causal: int = 48,
        num_graph_layers: int = 2,
        dropout: float = 0.1,
        edge_prior: float = -2.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_causal = d_causal

        self.encoder = CARTEStateEncoder(d_model, d_causal, dropout)
        self.graph = CARTECausalGraph(d_causal, num_graph_layers, dropout, edge_prior)

        # Input-dependent edge predictor: v → adj matrix
        self.edge_predictor = nn.Linear(d_causal, d_causal * d_causal)
        # Scale down initial edge predictions
        nn.init.normal_(self.edge_predictor.weight, std=0.01)
        nn.init.zeros_(self.edge_predictor.bias)
        self.edge_scale = 0.1

        # Decoder: project back to d_model
        self.decoder = nn.Linear(d_causal, d_model)

        # Learnable gate initialized at 0 (sigmoid(0) = 0.5, but we want ~0 at start)
        self.gate = nn.Parameter(torch.tensor(-5.0))  # sigmoid(-5) ≈ 0.007

        # Cached state from last forward
        self._last_edge_adj = None
        self._last_temperature = 1.0

    def forward(
        self,
        z_L: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        """
        Inject causal graph information into z_L.

        Args:
            z_L: [B, S, D] latent tensor
            step: current recursion step (0-indexed)
            total_steps: total recursion steps (T * n)
        Returns:
            z_L: [B, S, D] updated with causal injection
        """
        B, S, D = z_L.shape

        # Temperature annealing: 1.0 → 0.1 across steps
        t = step / max(total_steps - 1, 1)
        temperature = 1.0 - 0.9 * t
        self._last_temperature = temperature

        # 1. Pool z_L → last token
        z_pooled = z_L[:, -1, :]  # [B, D]

        # 2. Encode to causal space (in float32)
        with torch.amp.autocast("cuda", enabled=False):
            z_pooled_f32 = z_pooled.float()
            v = self.encoder(z_pooled_f32)  # [B, d_causal]

            # 3. Input-dependent edge prediction
            edge_flat = self.edge_predictor(v) * self.edge_scale  # [B, d*d]
            edge_adj = edge_flat.view(B, self.d_causal, self.d_causal)  # [B, d, d]
            self._last_edge_adj = edge_adj

            # 4. Graph message passing
            v_prime = self.graph(v, edge_adj, temperature=temperature)  # [B, d_causal]

            # 5. Decode + gated injection
            decoded = self.decoder(v_prime)  # [B, D]

        gate = torch.sigmoid(self.gate)
        z_L = z_L + gate * decoded.unsqueeze(1).to(z_L.dtype)

        return z_L

    def get_losses(self) -> dict:
        """Return causal regularization losses from last forward pass."""
        losses = {}
        temp = self._last_temperature
        with torch.amp.autocast("cuda", enabled=False):
            losses["acyclic"] = self.graph.acyclicity_loss(self._last_edge_adj, temperature=temp)
            losses["sparse"] = self.graph.sparsity_loss(self._last_edge_adj, temperature=temp)
            losses["ortho"] = self.encoder.orthogonality_loss()
        return losses

    def get_diagnostics(self) -> dict:
        stats = self.graph.get_graph_stats()
        stats["gate"] = round(torch.sigmoid(self.gate).item(), 4)
        return stats
