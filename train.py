"""
CARTE training script.

Dual optimizer: AdamW (main) + SignSGD (puzzle embeddings at 10x lr).
Saves init checkpoint for OrthoMerge before training starts.
"""

import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from carte import CARTEModel, CARTELoss, EMA
from datasets.puzzle_dataset import PuzzleDataset


def load_config(config_path: str, arch_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(arch_path) as f:
        arch = yaml.safe_load(f)
    cfg.update(arch)
    return cfg


def build_model(cfg: dict, device: str) -> CARTEModel:
    mc = cfg["model"]
    model = CARTEModel(
        vocab_size=mc["vocab_size"],
        d_model=mc["d_model"],
        n_heads=mc["n_heads"],
        expansion=mc["expansion"],
        T=mc["T"],
        n=mc["n"],
        d_causal=mc["d_causal"],
        max_puzzles=mc["max_puzzles"],
        max_seq_len=mc["max_seq_len"],
        dropout=mc["dropout"],
        num_graph_layers=mc["num_graph_layers"],
        edge_prior=mc["edge_prior"],
    )
    return model.to(device)


def build_optimizers(model: CARTEModel, cfg: dict):
    tc = cfg["training"]

    # Split params: puzzle embeddings vs everything else
    puzzle_params = []
    main_params = []
    for name, param in model.named_parameters():
        if "puzzle_emb" in name:
            puzzle_params.append(param)
        else:
            main_params.append(param)

    # Main optimizer: AdamW
    main_opt = torch.optim.AdamW(
        main_params,
        lr=tc["lr"],
        weight_decay=tc["weight_decay"],
        betas=(0.9, 0.999),
    )

    # Puzzle embedding optimizer: SignSGD at 10x lr
    puzzle_opt = None
    if puzzle_params:
        puzzle_opt = torch.optim.SGD(
            puzzle_params,
            lr=tc["lr"] * tc["puzzle_lr_mult"],
        )

    return main_opt, puzzle_opt


def train(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc = cfg["training"]
    lc = cfg["loss"]
    dc = cfg["data"]
    logc = cfg["logging"]
    cpc = cfg["checkpoint"]

    torch.manual_seed(tc["seed"])

    # Build model
    model = build_model(cfg, device)
    params = model.count_parameters()
    print(f"CARTE Model: {params['total']:,} params "
          f"(TRM: {params['base_trm']:,}, CausalTrack: {params['causal_track']:,} = {params['causal_pct']}%)")

    # Save init checkpoint for OrthoMerge
    save_dir = Path(cpc["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    if cpc.get("save_init", True):
        init_path = save_dir / "carte_init.pt"
        torch.save({"model_state_dict": model.state_dict()}, init_path)
        print(f"Saved init checkpoint: {init_path}")

    # Dataset
    train_dataset = PuzzleDataset(dc["data_dir"], split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=dc["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = PuzzleDataset(dc["data_dir"], split="val")
    val_loader = DataLoader(val_dataset, batch_size=tc["batch_size"], shuffle=False)

    # Optimizers
    main_opt, puzzle_opt = build_optimizers(model, cfg)

    # Scheduler: cosine with warmup
    def lr_lambda(step):
        if step < tc["warmup_steps"]:
            return step / tc["warmup_steps"]
        progress = (step - tc["warmup_steps"]) / (tc["max_steps"] - tc["warmup_steps"])
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(main_opt, lr_lambda)

    # Loss
    criterion = CARTELoss(
        lambda_acyclic=lc["lambda_acyclic"],
        lambda_sparse=lc["lambda_sparse"],
        lambda_ortho=lc["lambda_ortho"],
        halt_weight=lc["halt_weight"],
        ramp_fraction=lc["ramp_fraction"],
    )

    # EMA
    ema = EMA(model, decay=tc["ema_decay"])

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if tc.get("bf16") and device == "cuda" else None

    # Wandb
    if logc.get("wandb"):
        import wandb
        wandb.init(project=logc["wandb_project"], config=cfg)

    # Training loop
    global_step = 0
    data_iter = iter(train_loader)
    model.train()

    print(f"\nStarting training for {tc['max_steps']} steps...")
    t0 = time.time()

    while global_step < tc["max_steps"]:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        halt_targets = batch["halt_targets"].to(device)
        puzzle_ids = batch["puzzle_id"].to(device)

        # Forward
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bool(scaler)):
            logits, halt_probs, aux = model(input_ids, puzzle_ids)
            loss, loss_dict = criterion(
                logits, targets, halt_probs, halt_targets,
                aux, global_step, tc["max_steps"],
            )
            loss = loss / tc["grad_accum"]

        # Backward
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (global_step + 1) % tc["grad_accum"] == 0:
            if scaler:
                scaler.unscale_(main_opt)

            nn.utils.clip_grad_norm_(model.parameters(), tc["max_grad_norm"])

            if scaler:
                scaler.step(main_opt)
                if puzzle_opt:
                    scaler.step(puzzle_opt)
                scaler.update()
            else:
                main_opt.step()
                if puzzle_opt:
                    # SignSGD: use sign of gradient
                    for p in puzzle_opt.param_groups[0]["params"]:
                        if p.grad is not None:
                            p.grad.data = p.grad.data.sign()
                    puzzle_opt.step()

            main_opt.zero_grad()
            if puzzle_opt:
                puzzle_opt.zero_grad()

            scheduler.step()
            ema.update(model)

        # Logging
        if global_step % logc["log_every"] == 0:
            elapsed = time.time() - t0
            steps_per_sec = (global_step + 1) / elapsed if elapsed > 0 else 0
            lr = scheduler.get_last_lr()[0]

            diag = aux.get("diagnostics", {})
            print(
                f"step {global_step:6d} | loss {loss_dict['total']:.4f} | "
                f"lm {loss_dict['lm']:.4f} | acyc {loss_dict['acyclic']:.4f} | "
                f"gate {diag.get('gate', 0):.4f} | edges {diag.get('hard_edges', '?')} | "
                f"lr {lr:.2e} | {steps_per_sec:.1f} steps/s"
            )

            if logc.get("wandb"):
                import wandb
                wandb.log({**loss_dict, **diag, "lr": lr, "step": global_step})

        # Eval
        if global_step > 0 and global_step % logc["eval_every"] == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vinp = vbatch["input_ids"].to(device)
                    vtgt = vbatch["targets"].to(device)
                    vhalt = vbatch["halt_targets"].to(device)
                    vpuz = vbatch["puzzle_id"].to(device)
                    vlogits, vhalt_probs, vaux = model(vinp, vpuz)
                    vloss, vld = criterion(
                        vlogits, vtgt, vhalt_probs, vhalt,
                        vaux, global_step, tc["max_steps"],
                    )
                    val_losses.append(vld["total"])
                    if len(val_losses) >= 20:
                        break

            val_loss = sum(val_losses) / len(val_losses)
            print(f"  [EVAL] step {global_step} | val_loss {val_loss:.4f}")
            model.train()

        # Save
        if global_step > 0 and global_step % logc["save_every"] == 0:
            ckpt = {
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "main_opt": main_opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": global_step,
                "config": cfg,
            }
            path = save_dir / f"carte_step{global_step}.pt"
            torch.save(ckpt, path)
            print(f"  Saved checkpoint: {path}")

        global_step += 1

    # Final save
    final_path = save_dir / "carte_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "step": global_step,
        "config": cfg,
    }, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/cfg_train.yaml")
    parser.add_argument("--arch", default="config/arch/carte.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config, args.arch)
    train(cfg)
