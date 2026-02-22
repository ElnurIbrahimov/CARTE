"""
CARTEModel: Main model with TRM recursive loop + CausalTrack injection.

Architecture:
- Token embedding + puzzle embedding
- L_level (shared ReasoningModule) for both z_L and z_H updates
- T outer cycles (first T-1 no_grad), n inner steps per cycle
- CausalTrack injected between z_L and z_H updates
- Q-head for ACT halting, output head for vocab logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .layers import ReasoningModule, RMSNorm
from .causal_track import CausalTrack


class CARTEModel(nn.Module):

    def __init__(
        self,
        vocab_size: int = 32,
        d_model: int = 512,
        n_heads: int = 8,
        expansion: int = 4,
        T: int = 3,
        n: int = 6,
        d_causal: int = 48,
        max_puzzles: int = 10000,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        num_graph_layers: int = 2,
        edge_prior: float = -2.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.T = T
        self.n = n
        self.total_steps = T * n

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.puzzle_emb = nn.Embedding(max_puzzles, d_model)

        # Shared reasoning module (L_level) — 2 transformer blocks
        self.L_level = ReasoningModule(d_model, n_heads, expansion, dropout)

        # CausalTrack — the CARTE innovation
        self.causal_track = CausalTrack(
            d_model, d_causal, num_graph_layers, dropout, edge_prior
        )

        # Output heads
        self.out_norm = RMSNorm(d_model)
        self.out_head = nn.Linear(d_model, vocab_size, bias=False)

        # ACT halting head: scalar per position (returns logits, not probs)
        self.q_head = nn.Linear(d_model, 1)

        # Init
        self.apply(self._init_weights)
        # Tie output embedding
        self.out_head.weight = self.tok_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            input_ids: [B, S] token indices
            puzzle_ids: [B] puzzle indices for learned embeddings
        Returns:
            logits: [B, S, vocab_size]
            halt_logits: [B, S] raw halt logits (pre-sigmoid)
            aux: dict with causal losses and diagnostics
        """
        B, S = input_ids.shape

        # Input embedding
        input_emb = self.tok_emb(input_ids)  # [B, S, D]
        if puzzle_ids is not None:
            puz_ids_clamped = puzzle_ids % self.puzzle_emb.num_embeddings
            puz_emb = self.puzzle_emb(puz_ids_clamped).unsqueeze(1)  # [B, 1, D]
            input_emb = input_emb + puz_emb

        # Initialize latent spaces
        z_L = torch.zeros_like(input_emb)
        z_H = torch.zeros_like(input_emb)

        global_step = 0

        for cycle in range(self.T):
            use_grad = (cycle == self.T - 1)  # Only last cycle has gradients

            for step in range(self.n):
                if use_grad:
                    z_L = self.L_level(z_L, z_H + input_emb)
                    z_L = self.causal_track(z_L, global_step, self.total_steps)
                    z_H = self.L_level(z_H, z_L)
                else:
                    with torch.no_grad():
                        z_L = self.L_level(z_L, z_H + input_emb)
                        z_L = self.causal_track(z_L, global_step, self.total_steps)
                        z_H = self.L_level(z_H, z_L)

                global_step += 1

        # Output
        z_out = self.out_norm(z_H)
        logits = self.out_head(z_out)
        halt_logits = self.q_head(z_out).squeeze(-1)

        # Collect causal losses
        aux = self.causal_track.get_losses()
        aux["diagnostics"] = self.causal_track.get_diagnostics()

        return logits, halt_logits, aux

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        causal = sum(p.numel() for p in self.causal_track.parameters())
        base = total - causal
        return {
            "total": total,
            "base_trm": base,
            "causal_track": causal,
            "causal_pct": round(100 * causal / total, 1),
        }
