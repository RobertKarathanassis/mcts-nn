#!/usr/bin/env python
"""
trainer.py
==========

Consumes self-play games, trains JohnNet, and checkpoints regularly.

Run:
    python -m scripts.trainer --gpu 0
"""
from __future__ import annotations
import argparse, pathlib, time, random, gzip, pickle, glob, os
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from network.network import JohnNet
from data import move_encoder as enc

# ─── paths ──────────────────────────────────────────────────────────────
ROOT        = pathlib.Path(__file__).resolve().parents[1] / "run"
SELFPLAY_D  = ROOT / "selfplay"
CKPT_D      = ROOT / "checkpoints"
BEST_NET    = ROOT / "best.pth"

CKPT_D.mkdir(parents=True, exist_ok=True)

# ─── hyper-params ───────────────────────────────────────────────────────
BUFFER_CAP          = 200_000     # positions
BATCH_SIZE          = 256
LR_INITIAL          = 1e-3
WD                 = 1e-4
CHECKPOINT_EVERY_G  = 500          # games
DEVICE_DEFAULT      = "cuda:0"

# ─── replay buffer ──────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.states   = deque(maxlen=capacity)   # torch tensors (C,8,8)
        self.pis      = deque(maxlen=capacity)   # torch tensors (4672,)
        self.zs       = deque(maxlen=capacity)   # torch scalars
    def __len__(self) -> int: return len(self.states)

    def add_game(self, npz_file: pathlib.Path) -> None:
        data = np.load(npz_file)
        s, p, z = data["s"], data["p"], data["z"]
        for si, pi, zi in zip(s, p, z):
            self.states.append(torch.from_numpy(si))
            self.pis.append(torch.from_numpy(pi))
            self.zs.append(torch.tensor(zi, dtype=torch.float32))

    def sample(self, batch_size: int, device: torch.device
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idxs = random.sample(range(len(self)), batch_size)
        s = torch.stack([self.states[i] for i in idxs]).to(device, non_blocking=True)
        p = torch.stack([self.pis[i]    for i in idxs]).to(device, non_blocking=True)
        z = torch.stack([self.zs[i]     for i in idxs]).to(device, non_blocking=True)
        return s, p, z

# ─── loss helpers ───────────────────────────────────────────────────────
def policy_loss(logits: torch.Tensor, target_pi: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=1)
    return -(target_pi * log_p).sum(dim=1).mean()

def value_loss(pred_v: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_v.squeeze(-1), target_z)

# ─── main trainer loop ──────────────────────────────────────────────────
def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--gpu", type=int, default=0)
    args = pa.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")

    net = JohnNet().to(device)
    if BEST_NET.exists():
        net.load_state_dict(torch.load(BEST_NET, map_location=device))
        print(f"Loaded best network {BEST_NET.name}")

    # weight-decay only on non-BN parameters
    decay, no_decay = [], []
    for n, p in net.named_parameters():
        (decay if ("bn" not in n and p.dim() > 1) else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": WD},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LR_INITIAL)

    buf = ReplayBuffer(BUFFER_CAP)
    seen_files: set[str] = set()
    games_since_ckpt = 0
    step = 0

    while True:
        # ── ingest new games ────────────────────────────────────────────
        new_files = [f for f in SELFPLAY_D.glob("*.npz") if f.name not in seen_files]
        for f in new_files:
            buf.add_game(f)
            seen_files.add(f.name)
            games_since_ckpt += 1
        if not new_files:
            time.sleep(3)      # wait for workers
            continue

        # ── train if we have enough data ───────────────────────────────
        if len(buf) < BATCH_SIZE:            # warm-up
            continue

        net.train()
        s_batch, p_batch, z_batch = buf.sample(BATCH_SIZE, device)
        logits, v_pred = net(s_batch)
        loss = value_loss(v_pred, z_batch) + policy_loss(logits, p_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        step += 1

        print(f"step {step:06d}  loss {loss.item():.4f}  buffer {len(buf)}", end="\r")

        # ── checkpoint ─────────────────────────────────────────────────
        if games_since_ckpt >= CHECKPOINT_EVERY_G:
            ckpt_path = CKPT_D / f"net_step_{step:07d}.pth"
            torch.save(net.state_dict(), ckpt_path)
            torch.save(net.state_dict(), BEST_NET)   # naïvely promote; arena will refine later
            print(f"\nSaved checkpoint {ckpt_path.name}")
            games_since_ckpt = 0
            net.eval()   # search uses eval-mode weights

if __name__ == "__main__":
    main()
