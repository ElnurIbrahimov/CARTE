# CARTE: Causal Recursive Tiny Engine

## What This Is

A hybrid architecture combining Samsung TRM (recursive tiny models), Causeway (Elnur's causal DAG learning), and OrthoMerge (Riemannian manifold merging). The novel contribution: a recursive tiny network that refines an explicit causal graph at each recursion step. Nobody has done this before.

## Architecture

TRM recurses a shared 2-layer transformer (ReasoningModule) to refine z_L (scratchpad) and z_H (answer). CARTE injects a **CausalTrack** between z_L and z_H updates — each step refines a sparse causal graph alongside the answer.

```
for cycle in range(T=3):           # outer cycles (first 2 no_grad)
  for step in range(n=6):          # inner steps
    z_L = L_level(z_L, z_H + input_emb)
    z_L = causal_track(z_L, step)   # ← the innovation
    z_H = L_level(z_H, z_L)
```

CausalTrack per step: pool → encode (QR rotation) → predict input-dependent edges → graph message passing → gated inject back into z_L. Temperature anneals 1.0→0.1 across steps.

## Key Files

| File | What |
|---|---|
| `carte/model.py` | CARTEModel — recursive loop + embeddings + output heads |
| `carte/causal_track.py` | Core innovation — StateEncoder, CausalGraph, CausalTrack orchestrator |
| `carte/layers.py` | TRM building blocks — RMSNorm, SwiGLU, RoPE, Attention, Block, ReasoningModule |
| `carte/losses.py` | Stablemax CE + halt BCE + causal regularization (acyclicity ramp) |
| `carte/ema.py` | EMA helper |
| `merging/orthomerge.py` | Procrustes + Lie algebra merge on O(n) manifold |
| `train.py` | Training script (dual optimizer, bf16 AMP, wandb) |
| `evaluate.py` | Eval with TTA voting |
| `merge.py` | OrthoMerge CLI |

## Model Stats

- 13.7M total params (d_model=512, 8 heads, SwiGLU expansion=4)
- CausalTrack: 195K params = 1.4% overhead
- Vocab: 13 tokens for ARC (0-9 colors + row_sep + grid_sep + pad), 11 for Sudoku

## Causeway Lineage

CausalTrack adapts from `C:\Users\asus\Desktop\causeway\causeway\`:
- `state_encoder.py` → `CARTEStateEncoder` (QR rotation, slimmed to 1-layer MLP)
- `causal_graph.py` → `CARTECausalGraph` (Gumbel-sigmoid, NOTEARS, added batched input-dependent edges)

NOT used from Causeway: InterventionEngine, DeltaPredictor, AsymmetricMSE, ConfidenceLoss.

## Training Plan ($34 RunPod)

1. Build Sudoku data: `python datasets/build_sudoku.py`
2. Train Sudoku: `python train.py --arch config/arch/carte_sudoku.yaml` (~8h RTX 4090, ~$6)
3. Build ARC data: `python datasets/build_arc.py --arc_dir ARC-AGI`
4. Train ARC: `python train.py --arch config/arch/carte.yaml` (~12h, ~$9)
5. Merge: `python merge.py --init checkpoints/carte_init.pt --specialists checkpoints/carte_sudoku_final.pt checkpoints/carte_arc_final.pt --output checkpoints/carte_merged.pt`
6. Eval: `python evaluate.py --checkpoint checkpoints/carte_merged.pt --data_dir data/arc --tta 5`

## OrthoMerge Math

Causeway's rotation matrix R lives on O(n). OrthoMerge operates in the Lie algebra of O(n). Same manifold — OrthoMerge is the only mathematically correct way to merge these models.

Steps: task vectors → Procrustes decompose (SVD → Q + R) → average Q's via logm/expm in Lie algebra → average residuals → reconstruct.

## DO NOT run on local GPU (RTX 4060) — laptop will crash. RunPod only.
