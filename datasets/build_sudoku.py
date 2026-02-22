"""
Build Sudoku-Extreme dataset from HuggingFace sapientinc/sudoku-extreme.
Tokenizes digits 0-9 (0=empty), saves as memory-mapped .npy files.
"""

import argparse
import numpy as np
from pathlib import Path


def parse_sudoku_string(s: str) -> np.ndarray:
    """Convert '003020600...' to [0,0,3,0,2,0,6,0,0,...] array of length 81."""
    return np.array([int(c) for c in s if c.isdigit()], dtype=np.int64)


def build_dataset(output_dir: str, max_samples: int = 100000):
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading sapientinc/sudoku-extreme from HuggingFace (streaming)...")
    ds = load_dataset("sapientinc/sudoku-extreme", split="train", streaming=True)

    inputs = []
    targets = []
    count = 0

    for row in ds:
        if count >= max_samples:
            break

        puzzle = row.get("puzzle") or row.get("input") or row.get("question")
        solution = row.get("solution") or row.get("output") or row.get("answer")

        if puzzle is None or solution is None:
            continue

        inp = parse_sudoku_string(str(puzzle))
        tgt = parse_sudoku_string(str(solution))

        if len(inp) == 81 and len(tgt) == 81:
            inputs.append(inp)
            targets.append(tgt)
            count += 1

    inputs = np.stack(inputs)
    targets = np.stack(targets)

    # Split 90/10
    n = len(inputs)
    n_train = int(n * 0.9)

    np.save(output_dir / "train_inputs.npy", inputs[:n_train])
    np.save(output_dir / "train_targets.npy", targets[:n_train])
    np.save(output_dir / "val_inputs.npy", inputs[n_train:])
    np.save(output_dir / "val_targets.npy", targets[n_train:])

    print(f"Saved {n_train} train, {n - n_train} val samples to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/sudoku")
    parser.add_argument("--max_samples", type=int, default=100000)
    args = parser.parse_args()
    build_dataset(args.output_dir, args.max_samples)
