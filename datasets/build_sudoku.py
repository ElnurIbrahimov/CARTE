"""
Build Sudoku-Extreme dataset from HuggingFace sapientinc/sudoku-extreme.
Tokenizes digits 0-9 (0=empty), saves as memory-mapped .npy files.
"""

import argparse
import csv
import numpy as np
from pathlib import Path


def parse_sudoku_string(s: str) -> np.ndarray:
    """Convert '003020600...' to [0,0,3,0,2,0,6,0,0,...] array of length 81."""
    return np.array([int(c) for c in s if c.isdigit()], dtype=np.int64)


def build_dataset(output_dir: str, max_samples: int = 100000):
    from huggingface_hub import hf_hub_download

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading sapientinc/sudoku-extreme from HuggingFace...")
    csv_path = hf_hub_download(
        "sapientinc/sudoku-extreme", "train.csv", repo_type="dataset"
    )

    print(f"Parsing {csv_path}...")
    inputs = []
    targets = []
    count = 0

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        print(f"Columns: {reader.fieldnames}")

        for row in reader:
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

            if count % 10000 == 0 and count > 0:
                print(f"  Processed {count} samples...")

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
