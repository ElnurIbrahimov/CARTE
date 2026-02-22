"""
OrthoMerge specialist models.

Usage:
    python merge.py --init checkpoints/carte_init.pt \
                    --specialists checkpoints/carte_sudoku_final.pt checkpoints/carte_arc_final.pt \
                    --output checkpoints/carte_merged.pt
"""

import argparse
from merging.orthomerge import merge_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OrthoMerge CARTE specialists")
    parser.add_argument("--init", required=True, help="Path to init checkpoint")
    parser.add_argument("--specialists", nargs="+", required=True, help="Paths to trained specialist checkpoints")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    parser.add_argument("--merge_embeddings", action="store_true", help="Also merge embedding layers")
    args = parser.parse_args()

    merge_models(
        init_checkpoint=args.init,
        specialist_checkpoints=args.specialists,
        output_path=args.output,
        merge_embeddings=args.merge_embeddings,
    )
