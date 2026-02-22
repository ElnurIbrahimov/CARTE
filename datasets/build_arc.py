"""
Build ARC-AGI dataset for CARTE training.
Tokenizes grid cells (0-9 colors) + special tokens for grid structure.
"""

import argparse
import json
import numpy as np
from pathlib import Path


# Token vocabulary for ARC:
# 0-9: colors, 10: row separator, 11: grid separator, 12: padding
VOCAB_SIZE = 13
ROW_SEP = 10
GRID_SEP = 11
PAD = 12


def grid_to_tokens(grid: list) -> list:
    """Convert ARC grid to flat token sequence with row separators."""
    tokens = []
    for i, row in enumerate(grid):
        tokens.extend(row)
        if i < len(grid) - 1:
            tokens.append(ROW_SEP)
    return tokens


def encode_task(task: dict, max_len: int = 512) -> tuple:
    """
    Encode an ARC task as (input_tokens, target_tokens).
    Format: [demo_in GRID_SEP demo_out GRID_SEP ... test_in]
    Target: [PAD... test_out]
    """
    context = []
    for demo in task["train"]:
        context.extend(grid_to_tokens(demo["input"]))
        context.append(GRID_SEP)
        context.extend(grid_to_tokens(demo["output"]))
        context.append(GRID_SEP)

    test_in = grid_to_tokens(task["test"][0]["input"])
    context.extend(test_in)

    test_out = grid_to_tokens(task["test"][0]["output"])

    # Pad/truncate
    input_tokens = context[:max_len]
    target_tokens = test_out[:max_len]

    input_tokens = input_tokens + [PAD] * (max_len - len(input_tokens))
    target_tokens = target_tokens + [PAD] * (max_len - len(target_tokens))

    return np.array(input_tokens, dtype=np.int64), np.array(target_tokens, dtype=np.int64)


def build_dataset(arc_dir: str, output_dir: str, max_len: int = 512):
    arc_dir = Path(arc_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = []
    targets = []

    # Training tasks
    train_dir = arc_dir / "training"
    if not train_dir.exists():
        train_dir = arc_dir / "data" / "training"

    if not train_dir.exists():
        print(f"ARC training dir not found at {train_dir}. Download ARC-AGI first.")
        print("git clone https://github.com/fchollet/ARC-AGI.git")
        return

    for task_file in sorted(train_dir.glob("*.json")):
        with open(task_file) as f:
            task = json.load(f)
        inp, tgt = encode_task(task, max_len)
        inputs.append(inp)
        targets.append(tgt)

    # Evaluation tasks
    eval_dir = arc_dir / "evaluation"
    if not eval_dir.exists():
        eval_dir = arc_dir / "data" / "evaluation"

    if eval_dir.exists():
        eval_inputs = []
        eval_targets = []
        for task_file in sorted(eval_dir.glob("*.json")):
            with open(task_file) as f:
                task = json.load(f)
            inp, tgt = encode_task(task, max_len)
            eval_inputs.append(inp)
            eval_targets.append(tgt)

        np.save(output_dir / "eval_inputs.npy", np.stack(eval_inputs))
        np.save(output_dir / "eval_targets.npy", np.stack(eval_targets))
        print(f"Saved {len(eval_inputs)} eval samples")

    inputs = np.stack(inputs)
    targets = np.stack(targets)

    np.save(output_dir / "train_inputs.npy", inputs)
    np.save(output_dir / "train_targets.npy", targets)
    print(f"Saved {len(inputs)} train samples to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc_dir", default="ARC-AGI")
    parser.add_argument("--output_dir", default="data/arc")
    parser.add_argument("--max_len", type=int, default=512)
    args = parser.parse_args()
    build_dataset(args.arc_dir, args.output_dir, args.max_len)
