"""
Only chatGPT and god know whats going on here. Goodluck 
"""

from __future__ import annotations
import numpy as np
import chess
from typing import Optional

# ── spec constants ────────────────────────────────────────────────────
NUM_QUEEN_DIRS, NUM_QUEEN_DISTS = 8, 7
NUM_QUEEN_MOVES   = NUM_QUEEN_DIRS * NUM_QUEEN_DISTS          # 56
NUM_KNIGHT_MOVES  = 8
NUM_UNDERPROMO    = 3 * 3                                     #  9
MOVE_TYPE_TOTAL   = NUM_QUEEN_MOVES + NUM_KNIGHT_MOVES + NUM_UNDERPROMO
ACTION_SIZE       = 8 * 8 * MOVE_TYPE_TOTAL                   # 4 672

QUEEN_OFFSET      = 0
KNIGHT_OFFSET     = QUEEN_OFFSET   + NUM_QUEEN_MOVES          # 56
UNDERPROMO_OFFSET = KNIGHT_OFFSET  + NUM_KNIGHT_MOVES         # 64

QUEEN_DIRS: list[tuple[int, int]] = [
    ( 1,  0), ( 1,  1), ( 0,  1), (-1,  1),
    (-1,  0), (-1, -1), ( 0, -1), ( 1, -1),
]
KNIGHT_DELTAS = [
    ( 2,  1), ( 1,  2), (-1,  2), (-2,  1),
    (-2, -1), (-1, -2), ( 1, -2), ( 2, -1),
]
UNDERPROMO_DELTAS = [-1, 0, 1]                       # L-capture / push / R-capture
UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

FEN_SUITE = {
    "castling_fest":      "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "white_en_passant":   "rnbqkbnr/ppp1pppp/8/3p4/2P5/8/PP1PPPPP/RNBQKBNR w KQkq d6 0 2",
    "black_en_passant":   "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2",
    "white_promos_full" : "8/P7/8/8/8/8/8/8 w - - 0 1",   # white: 4 moves
    "black_promos_full" : "8/8/8/8/8/8/p7/8 b - - 0 1"  # black: 4 moves
}

mirror_if_black = lambda sq, colour: chess.square_mirror(sq) if colour == chess.BLACK else sq


# ── encoder ────────────────────────────────────────────────────────────
def encode_move(board: chess.Board, move: chess.Move) -> Optional[int]:
    """Return αZero index [0, 4671] for `move`, or None if unsupported."""
    side     = board.turn
    from_sq  = mirror_if_black(move.from_square, side)
    to_sq    = mirror_if_black(move.to_square,   side)
    fr, ff   = divmod(from_sq, 8)
    tr, tf   = divmod(to_sq,   8)
    dy, dx   = tr - fr, tf - ff

    # Queen-style rays (incl. king, rook, bishop, pawn pushes/captures, castles, *queen-promos*)
    if move.promotion in (None, chess.QUEEN):
        dir_vec = (int(np.sign(dy)), int(np.sign(dx)))
        if dir_vec in QUEEN_DIRS:
            dist = max(abs(dy), abs(dx))
            if dy == dir_vec[0] * dist and dx == dir_vec[1] * dist and 1 <= dist <= 7:
                mtype = QUEEN_OFFSET + QUEEN_DIRS.index(dir_vec) * 7 + (dist - 1)
                return int(np.ravel_multi_index((fr, ff, mtype), (8, 8, MOVE_TYPE_TOTAL)))

    # Knight jumps
    if move.promotion is None and (dy, dx) in KNIGHT_DELTAS:
        mtype = KNIGHT_OFFSET + KNIGHT_DELTAS.index((dy, dx))
        return int(np.ravel_multi_index((fr, ff, mtype), (8, 8, MOVE_TYPE_TOTAL)))

    # Under-promos (to N/B/R)
    if (
        move.promotion in UNDERPROMO_PIECES
        and fr == 6 and dy == 1 and dx in UNDERPROMO_DELTAS
    ):
        mtype = (
            UNDERPROMO_OFFSET
            + UNDERPROMO_DELTAS.index(dx) * 3
            + UNDERPROMO_PIECES.index(move.promotion)
        )
        return int(np.ravel_multi_index((fr, ff, mtype), (8, 8, MOVE_TYPE_TOTAL)))

    return None


# ── decoder ────────────────────────────────────────────────────────────
def decode_index(board: chess.Board, index: int) -> chess.Move:
    """
    Inverse of `encode_move`.

    Parameters
    ----------
    board : chess.Board
        *Current* position – needed to decide if auto-promotion applies.
    index : int  (0 ≤ index < 4 672)
    """
    if not 0 <= index < ACTION_SIZE:
        raise ValueError("index must be in [0, 4672)")

    fr, ff, mtype = map(int, np.unravel_index(index, (8, 8, MOVE_TYPE_TOTAL)))
    from_sq = chess.square(ff, fr)

    if mtype < KNIGHT_OFFSET:                                 # queen-style
        rel   = mtype - QUEEN_OFFSET
        dir_i = rel // 7
        dist  = (rel % 7) + 1
        dy, dx  = QUEEN_DIRS[dir_i]
        to_sq   = chess.square(ff + dx * dist, fr + dy * dist)

        # auto-queen only if a pawn is making the advance
        promo = (
            chess.QUEEN
            if fr == 6 and dy == 1 and board.piece_type_at(mirror_if_black(from_sq, board.turn)) == chess.PAWN
            else None
        )

    elif mtype < UNDERPROMO_OFFSET:                           # knights
        dy, dx = KNIGHT_DELTAS[mtype - KNIGHT_OFFSET]
        to_sq  = chess.square(ff + dx, fr + dy)
        promo  = None

    else:                                                     # under-promos
        rel      = mtype - UNDERPROMO_OFFSET
        dx       = UNDERPROMO_DELTAS[rel // 3]
        promo    = UNDERPROMO_PIECES[rel % 3]
        to_sq    = chess.square(ff + dx, fr + 1)

    # un-mirror for Black
    from_sq = mirror_if_black(from_sq, board.turn)
    to_sq   = mirror_if_black(to_sq,   board.turn)

    return chess.Move(from_sq, to_sq, promotion=promo)

def run_suite() -> None:
    for name, fen in FEN_SUITE.items():
        board = chess.Board(fen)
        for mv in list(board.legal_moves):
            idx = encode_move(board, mv)
            assert idx is not None, f"[{name}] un-encodable {mv}"
            mv2 = decode_index(board, idx)
            assert mv == mv2, f"[{name}] mismatch {mv} ≠ {mv2}"
        print(f"{name}: ✅ {board.legal_moves.count()} moves round-tripped")


if __name__ == "__main__":
    import random, sys

    def fuzz(plies: int = 10_000) -> None:
        b = chess.Board()
        for _ in range(plies):
            for mv in list(b.legal_moves):
                idx = encode_move(b, mv)
                assert idx is not None, f"un-encodable {mv}"
                mv2 = decode_index(b, idx)
                assert mv == mv2, f"mismatch {mv} vs {mv2}"
            b.push(random.choice(list(b.legal_moves)))
            if b.is_game_over():
                b.reset()
        print(f"Fuzzed {plies} plies without error ✅")

    try:
        fuzz()
    except AssertionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    
    run_suite()
