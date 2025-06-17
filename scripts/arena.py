#!/usr/bin/env python
# path: scripts/arena.py

from __future__ import annotations
import argparse, csv, pathlib, os, time, subprocess
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import torch, chess
from mcts.mcts import MCTS
from network.network import JohnNet
from data import move_encoder as enc

ROOT_DIR       = pathlib.Path(__file__).resolve().parents[1] / "run"
BEST_NET_PATH  = ROOT_DIR / "best.pth"
CKPT_DIR       = ROOT_DIR / "checkpoints"
ARENA_LOG_PATH = ROOT_DIR / "arena_log.csv"

DEFAULT_GAMES  = 50
DEFAULT_SIMS   = 200
PROMOTE_THRESH = 0.55

@dataclass
class Score:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    def record(self, result: str, white: bool) -> None:
        if result == "1/2-1/2":
            self.draws += 1
        elif (result == "1-0") == white:
            self.wins += 1
        else:
            self.losses += 1
    @property
    def total(self) -> int: return self.wins + self.losses + self.draws
    @property
    def score_frac(self) -> float:
        return 0.0 if self.total == 0 else (self.wins + 0.5*self.draws) / self.total

def play_single_game(gid: int, gpu: int, sims: int,
                     best_sd: dict, new_sd: dict) -> Tuple[str, bool]:
    if gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda:0" if gpu >= 0 else "cpu")

    best_net = JohnNet().to(device); best_net.load_state_dict(best_sd); best_net.eval()
    new_net  = JohnNet().to(device); new_net.load_state_dict(new_sd); new_net.eval()

    new_white = gid % 2 == 0
    board = chess.Board()
    while not board.is_game_over():
        net = new_net if ((board.turn == chess.WHITE) == new_white) else best_net
        visits = MCTS(net, sims=sims).run(board)
        move_idx = max(visits, key=visits.get)
        board.push(enc.decode_index(board, move_idx))
    return board.result(), new_white

def run_arena(new_ckpt: pathlib.Path, games: int, sims: int, gpus: str) -> None:
    print(f"\n⏳ Arena vs best.pth | games={games} sims={sims}")
    if not BEST_NET_PATH.exists():
        print("⚠️  Skipping arena – best.pth not yet available.")
        return

    best_sd = torch.load(BEST_NET_PATH, map_location="cpu")
    new_sd  = torch.load(new_ckpt, map_location="cpu")

    gpu_ids = [int(x) for x in gpus.split(",")] if gpus else [-1]
    gpu_cycle = (gpu_ids * ((games + len(gpu_ids)-1)//len(gpu_ids)))[:games]

    score, t0 = Score(), time.time()
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futs = [pool.submit(play_single_game, gid, gpu_cycle[gid], sims, best_sd, new_sd)
                for gid in range(games)]
        for f in as_completed(futs):
            res, white = f.result()
            score.record(res, white)
            print(f"\r  finished {score.total}/{games} "
                  f"W{score.wins}-L{score.losses}-D{score.draws}", end="", flush=True)

    print(f"\n✅ Completed in {(time.time()-t0)/60:.1f} min | "
          f"score={score.score_frac*100:.1f}%")

    promoted = score.score_frac > PROMOTE_THRESH
    if promoted:
        torch.save(new_sd, BEST_NET_PATH)
        print("🎉 PROMOTED → best.pth updated")
    else:
        print("🚫 Not promoted")

    ARENA_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not ARENA_LOG_PATH.exists()
    with open(ARENA_LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp","checkpoint","games","wins","losses",
                        "draws","score_frac","promoted"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), new_ckpt.name, games,
                    score.wins, score.losses, score.draws,
                    round(score.score_frac, 4), int(promoted)])

def arena_watch_loop(games: int, sims: int, gpus: str):
    seen = set()
    print(f"🔁 Starting arena loop – watching {CKPT_DIR} for new .pth checkpoints…")
    while True:
        ckpts = sorted(CKPT_DIR.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            latest = ckpts[0]
            if latest.name not in seen:
                print(f"\n🧠 New checkpoint detected: {latest.name}")
                run_arena(latest, games, sims, gpus)
                seen.add(latest.name)
        time.sleep(10)

def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--new_ckpt", type=pathlib.Path)
    pa.add_argument("--games", type=int, default=DEFAULT_GAMES)
    pa.add_argument("--sims",  type=int, default=DEFAULT_SIMS)
    pa.add_argument("--gpus",  type=str, default="0")
    args = pa.parse_args()

    if args.new_ckpt is not None:
        run_arena(args.new_ckpt, args.games, args.sims, args.gpus)
    else:
        arena_watch_loop(args.games, args.sims, args.gpus)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
