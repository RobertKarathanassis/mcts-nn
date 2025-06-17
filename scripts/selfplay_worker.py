#!/usr/bin/env python
"""
Self-play worker
================
Generates (state, π, z) and stores them in run/selfplay/ as .npz.
Every restart continues numbering from the last existing file so
nothing is over-written.
"""
from __future__ import annotations
import argparse, pathlib, time, random, multiprocessing
import numpy as np
import torch, chess

from data import board, move_encoder as enc
from mcts.mcts import MCTS
from network.network import JohnNet

# ── runtime dirs ───────────────────────────────────────────────────────
ROOT          = pathlib.Path(__file__).resolve().parents[1] / "run"
SELFPLAY_DIR  = ROOT / "selfplay"
BEST_NET_PATH = ROOT / "best.pth"
SELFPLAY_DIR.mkdir(parents=True, exist_ok=True)

# ── hyper-params ───────────────────────────────────────────────────────
NUM_SIMS    = 30
TEMP_CUTOFF = 30
DIRICH_A    = 0.3
DIRICH_EPS  = 0.25

# ── figure out next unused game idx for a worker ───────────────────────
def next_game_idx(worker_id: int) -> int:
    pattern = f"w{worker_id}_g*.npz"
    existing = [
        int(p.stem.split("_g")[1])
        for p in SELFPLAY_DIR.glob(pattern)
    ]
    return max(existing, default=-1) + 1

# ── helpers ────────────────────────────────────────────────────────────
def sample_move(visits: dict[int, int], tau: float) -> int:
    idxs, counts = zip(*visits.items())
    counts = np.array(counts, dtype=np.float32)
    if tau == 0.0:
        return int(idxs[int(counts.argmax())])
    probs = counts ** (1 / tau)
    probs /= probs.sum()
    return int(np.random.choice(idxs, p=probs))

def result_to_z(result: str, pov: chess.Color) -> float:
    if result == "1-0":
        return  1.0 if pov == chess.WHITE else -1.0
    if result == "0-1":
        return -1.0 if pov == chess.WHITE else  1.0
    return 0.0

def load_net(device: torch.device) -> torch.nn.Module:
    net = JohnNet().to(device)
    if BEST_NET_PATH.exists():
        net.load_state_dict(torch.load(BEST_NET_PATH, map_location=device))
        net._loaded_ts = BEST_NET_PATH.stat().st_mtime  # type: ignore[attr-defined]
    else:
        net._loaded_ts = 0.0                             # type: ignore[attr-defined]
    net.eval()
    return net

# ── self-play one game ────────────────────────────────────────────────
def play_game(worker_id: int, game_id: int, device: torch.device) -> None:
    net  = load_net(device)
    tree = MCTS(net, sims=NUM_SIMS, dir_alpha=DIRICH_A, dir_eps=DIRICH_EPS)

    b = chess.Board()
    states, pis, turns = [], [], []
    ply = 0

    while not b.is_game_over():
        visits = tree.run(b)
        tau = 1.0 if ply < TEMP_CUTOFF else 0.0
        move_idx = sample_move(visits, tau)

        states.append(board.board_to_tensor(b))
        pi = np.zeros(enc.ACTION_SIZE, dtype=np.float32)
        for k, n in visits.items():
            pi[k] = n
        pi /= pi.sum()
        pis.append(pi)
        turns.append(b.turn)

        b.push(enc.decode_index(b, move_idx))
        ply += 1

        # hot-reload weights every 20 plies
        if (
            ply % 20 == 0
            and BEST_NET_PATH.exists()
            and BEST_NET_PATH.stat().st_mtime > getattr(net, "_loaded_ts")
        ):
            net = load_net(device)
            tree.net = net

    z_final = result_to_z(b.result(), turns[0])
    z = np.array([z_final if t == turns[0] else -z_final for t in turns],
                 dtype=np.float32)

    out = SELFPLAY_DIR / f"w{worker_id}_g{game_id:06d}.npz"
    np.savez_compressed(out, s=np.stack(states), p=np.stack(pis), z=z)
    print(f"[worker {worker_id}] saved {out.name}  plies={ply}")

# ── CLI entry-point ────────────────────────────────────────────────────
def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--games", type=int, default=10)
    pa.add_argument("--gpu",   type=int, default=-1)   # -1 = CPU
    pa.add_argument("--id",    type=int, default=0)    # worker id offset
    pa.add_argument("--workers", type=int, default=1)
    args = pa.parse_args()

    device_str = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"

    def run_worker(wid: int) -> None:
        device = torch.device(device_str)
        start = time.time()

        start_g = next_game_idx(wid)
        for g in range(start_g, start_g + args.games):
            play_game(wid, g, device)

        print(f"[worker {wid}] finished {args.games} games in {time.time()-start:.1f}s")

    procs = []
    for i in range(args.workers):
        wid = args.id + i
        p = multiprocessing.Process(target=run_worker, args=(wid,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

if __name__ == "__main__":
    main()
