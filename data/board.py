import torch
import chess
import numpy as np

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a python-chess board into a tensor.
    
    Output shape: (20, 8, 8)
    Channels:
    - 0 - 5:    White pieces (P, N, B, R, Q, K)
    - 6 - 11:   Black pieces (p, n, b, r, q, k)
    - 12:     Side to move (1 = white)
    - 13 - 16:  Castling rights (white K/Q, black K/Q)
    - 17:     En passant square (1 at target square)
    - 18:     Fifty-move clock (normalized)
    - 19:     Fullmove number (normalized)
    """

    tensor: np.ndarray = np.zeros((20, 8, 8), dtype=np.float32)

    # Piece placement
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        plane = piece_type_plane(piece)
        tensor[plane, row, col] = 1.0

    # Side to move (plane 12)
    tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # Castling rights
    tensor[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # En passant (plane 17) — mark the en passant square, if it exists
    if board.ep_square is not None:
        row = 7 - (board.ep_square // 8)
        col = board.ep_square % 8
        tensor[17, row, col] = 1.0

    # Fifty-move clock (plane 18) — normalized to [0, 1]
    tensor[18, :, :] = min(board.halfmove_clock / 100.0, 1.0)

    # Fullmove number (plane 19) — normalized arbitrarily to [0, 1]
    # TODO: Address the 300 magic number if neccesary 
    tensor[19, :, :] = min(board.fullmove_number / 300.0, 1.0)

    return torch.from_numpy(tensor)

def piece_type_plane(piece: chess.Piece) -> int:
    """
    Map a chess piece to the corresponding channel index in the tensor.

    - White pieces go in planes 0 - 5
    - Black pieces go in planes 6 - 11
    Plane index is determined by piece type (PAWN=0, KNIGHT=1, ..., KING=5)
    """
    offset = 0 if piece.color == chess.WHITE else 6
    return offset + {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }[piece.piece_type]