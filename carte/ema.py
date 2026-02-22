"""Exponential Moving Average of model parameters."""

import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    """Maintains shadow parameters with exponential moving average."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def ema_copy(self) -> nn.Module:
        """Return a copy of the shadow model for evaluation."""
        return deepcopy(self.shadow)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)
