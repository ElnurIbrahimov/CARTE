"""
Memory-mapped puzzle dataset for CARTE training.
Loads .npy files created by build_sudoku.py / build_arc.py.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class PuzzleDataset(Dataset):
    """Dataset backed by memory-mapped .npy files."""

    def __init__(self, data_dir: str, split: str = "train"):
        data_dir = Path(data_dir)
        self.inputs = np.load(data_dir / f"{split}_inputs.npy", mmap_mode="r")
        self.targets = np.load(data_dir / f"{split}_targets.npy", mmap_mode="r")
        assert len(self.inputs) == len(self.targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp = torch.from_numpy(self.inputs[idx].copy()).long()
        tgt = torch.from_numpy(self.targets[idx].copy()).long()

        # Halt target: 1 where target differs from input (puzzle cells to solve)
        halt_target = (inp != tgt).float()

        return {
            "input_ids": inp,
            "targets": tgt,
            "halt_targets": halt_target,
            "puzzle_id": idx,
        }
