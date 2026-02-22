"""
CARTE evaluation script.

Usage:
    python evaluate.py --checkpoint checkpoints/carte_final.pt \
                       --data_dir data/arc --tta 5
"""

import argparse
import yaml
import torch
from pathlib import Path

from carte import CARTEModel
from datasets.puzzle_dataset import PuzzleDataset
from evaluators.arc import ArcEvaluator


def load_model(checkpoint_path: str, device: str) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {})
    mc = cfg.get("model", {})

    model = CARTEModel(
        vocab_size=mc.get("vocab_size", 13),
        d_model=mc.get("d_model", 512),
        n_heads=mc.get("n_heads", 8),
        expansion=mc.get("expansion", 4),
        T=mc.get("T", 3),
        n=mc.get("n", 6),
        d_causal=mc.get("d_causal", 48),
        max_puzzles=mc.get("max_puzzles", 10000),
        max_seq_len=mc.get("max_seq_len", 512),
    )

    state_dict = ckpt.get("model_state_dict") or ckpt.get("ema_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--tta", type=int, default=0, help="TTA votes (0=disabled)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint, args.device)
    params = model.count_parameters()
    print(f"Model: {params['total']:,} params")

    dataset = PuzzleDataset(args.data_dir, split=args.split)
    print(f"Dataset: {len(dataset)} samples")

    evaluator = ArcEvaluator(model, args.device)

    if args.tta > 0:
        print(f"Evaluating with TTA (n_votes={args.tta})...")
        correct = 0
        total_cell_acc = 0.0
        n = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)

        for i in range(n):
            sample = dataset[i]
            result = evaluator.evaluate_with_tta(
                sample["input_ids"], sample["targets"], n_votes=args.tta
            )
            correct += result["correct"]
            total_cell_acc += result["cell_accuracy"]

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{n} | acc {correct/(i+1):.4f} | cell_acc {total_cell_acc/(i+1):.4f}")

        print(f"\nResults (TTA={args.tta}):")
        print(f"  Task accuracy: {correct/n:.4f} ({correct}/{n})")
        print(f"  Cell accuracy: {total_cell_acc/n:.4f}")
    else:
        print("Evaluating (no TTA)...")
        results = evaluator.evaluate_dataset(dataset, max_samples=args.max_samples)
        print(f"\nResults:")
        print(f"  Task accuracy: {results['task_accuracy']:.4f} ({results['n_correct']}/{results['n_tasks']})")
        print(f"  Cell accuracy: {results['cell_accuracy']:.4f}")


if __name__ == "__main__":
    main()
