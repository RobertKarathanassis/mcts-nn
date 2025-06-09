# mcts.py
from __future__ import annotations
import math, random
import numpy as np
import torch
import chess
from collections import defaultdict
from typing import Optional, Tuple, cast

import data.move_encoder as encoder     
from data.board import board_to_tensor

# ─── helper ─────────────────────────────────────────────────────────────
def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max()
    exp    = np.exp(logits)
    return exp / exp.sum()

# ─── node ───────────────────────────────────────────────────────────────
class Node:
    __slots__ = ("parent", "to_play", "prior", "n", "w", "children")
    def __init__(self,
                 parent   : Optional["Node"],
                 to_play  : chess.Color,
                 prior    : float):
        self.parent   = parent
        self.to_play  = to_play
        self.prior    = prior
        self.n        = 0          # visit count
        self.w        = 0.0        # total value from this player's POV
        self.children : dict[int, "Node"] = {}

    @property
    def q(self) -> float:
        return 0.0 if self.n == 0 else self.w / self.n

    def u(self, c_puct: float, n_parent: int) -> float:
        return c_puct * self.prior * math.sqrt(n_parent) / (1 + self.n)

# ─── mcts core ──────────────────────────────────────────────────────────
class MCTS:
    def __init__(self, net, c_puct=1.5,
                 dir_alpha=0.3, dir_eps=0.25, sims=800):
        self.net      = net
        self.c_puct   = c_puct
        self.dir_a    = dir_alpha
        self.dir_eps  = dir_eps
        self.sims     = sims

    # -------- public ----------------------------------------------------
    def run(self, board: chess.Board) -> dict[int, int]:
        root = Node(None, board.turn, 0.0)
        self._expand(root, board)              # populate root priors
        self._add_dirichlet_noise(root)

        for _ in range(self.sims):
            scratch = board.copy(stack=False)
            path    = [root]
            node    = root

            # Selection
            while node.children:
                idx, node = self._select_child(node)
                scratch.push(encoder.decode_index(scratch, idx))
                path.append(node)

            # Expansion & evaluation
            value = self._expand(node, scratch)

            # Back-prop
            for n in reversed(path):
                n.n += 1
                n.w += value if n.to_play == board.turn else -value

        # return visit counts for policy head training / move sampling
        return {idx: child.n for idx, child in root.children.items()}

    # -------- private ---------------------------------------------------
    def _select_child(self, node: Node) -> Tuple[int, Node]:
    # pick child maximising Q + U
        n_parent = node.n
        best_idx: Optional[int]  = None
        best_child: Optional[Node] = None
        best_score = -float("inf")

        for idx, child in node.children.items():
            score = child.q + child.u(self.c_puct, n_parent)
            if score > best_score:
                best_idx, best_child, best_score = idx, child, score

        # cannot be None because node.children is not empty
        assert best_idx is not None and best_child is not None
        return cast(Tuple[int, Node], (best_idx, best_child))

    def _expand(self, node: Node, board: chess.Board) -> float:
        if board.is_game_over():
            result = board.result()
            # draw = 0, win = 1, loss = –1 from POV of `node.to_play`
            if result == "1-0":
                return 1.0 if node.to_play == chess.WHITE else -1.0
            elif result == "0-1":
                return -1.0 if node.to_play == chess.WHITE else 1.0
            else:
                return 0.0

        # network inference
        planes = board_to_tensor(board)              # may be tensor or ndarray

        if isinstance(planes, torch.Tensor):         # already a tensor → no copy
            tensor = planes
        else:                                        # numpy → zero-copy view
            tensor = torch.from_numpy(planes)

        tensor = tensor.unsqueeze(0).to(
            next(self.net.parameters()).device, non_blocking=True
        )                                            # shape (1, C, 8, 8)

        with torch.no_grad():
            logits, value = self.net(tensor)
            logits = logits.squeeze().cpu().numpy()
            value  = value.item()

        # legal moves mask + softmax
        priors = np.zeros(encoder.ACTION_SIZE, dtype=np.float32)
        for mv in board.legal_moves:
            idx = encoder.encode_move(board, mv)
            priors[idx] = logits[idx]
        priors = softmax(priors)

        # populate children
        for mv in board.legal_moves:
            idx = encoder.encode_move(board, mv)
            assert idx is not None
            node.children[idx] = Node(node, not board.turn, priors[idx])

        return value  # from current player's POV

    def _add_dirichlet_noise(self, root: Node) -> None:
        if not root.children:
            return
        alpha = self.dir_a
        noise = np.random.dirichlet([alpha] * len(root.children))
        for (idx, child), n in zip(root.children.items(), noise):
            child.prior = (1 - self.dir_eps) * child.prior + self.dir_eps * n
