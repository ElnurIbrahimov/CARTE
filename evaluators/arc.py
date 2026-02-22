"""ARC evaluator with TTA (Test-Time Augmentation) voting."""

import torch
import numpy as np
from typing import Optional


class ArcEvaluator:
    """Evaluate CARTE on ARC tasks with optional TTA voting."""

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate_single(self, input_ids: torch.Tensor, target: torch.Tensor) -> dict:
        """Evaluate a single ARC task."""
        self.model.eval()
        input_ids = input_ids.unsqueeze(0).to(self.device)
        logits, halt_probs, aux = self.model(input_ids)

        preds = logits.argmax(dim=-1).squeeze(0).cpu()
        target = target.cpu()

        # Exact match on non-padding positions
        mask = target != 12  # PAD token
        correct = (preds[mask] == target[mask]).all().item()
        cell_acc = (preds[mask] == target[mask]).float().mean().item()

        return {
            "correct": correct,
            "cell_accuracy": cell_acc,
            "diagnostics": aux.get("diagnostics", {}),
        }

    @torch.no_grad()
    def evaluate_dataset(self, dataset, max_samples: Optional[int] = None) -> dict:
        """Evaluate on full dataset."""
        self.model.eval()
        n = min(len(dataset), max_samples) if max_samples else len(dataset)

        correct = 0
        total_cell_acc = 0.0

        for i in range(n):
            sample = dataset[i]
            result = self.evaluate_single(sample["input_ids"], sample["targets"])
            correct += result["correct"]
            total_cell_acc += result["cell_accuracy"]

        return {
            "task_accuracy": correct / n,
            "cell_accuracy": total_cell_acc / n,
            "n_tasks": n,
            "n_correct": correct,
        }

    @torch.no_grad()
    def evaluate_with_tta(
        self, input_ids: torch.Tensor, target: torch.Tensor, n_votes: int = 5
    ) -> dict:
        """TTA voting: run model multiple times, take majority vote per cell."""
        self.model.eval()
        votes = []

        for _ in range(n_votes):
            inp = input_ids.unsqueeze(0).to(self.device)
            logits, _, _ = self.model(inp)
            preds = logits.argmax(dim=-1).squeeze(0).cpu()
            votes.append(preds)

        # Majority vote
        votes = torch.stack(votes, dim=0)  # [n_votes, S]
        majority = torch.mode(votes, dim=0).values

        target = target.cpu()
        mask = target != 12
        correct = (majority[mask] == target[mask]).all().item()
        cell_acc = (majority[mask] == target[mask]).float().mean().item()

        return {
            "correct": correct,
            "cell_accuracy": cell_acc,
            "n_votes": n_votes,
        }
