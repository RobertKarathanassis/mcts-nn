try:
    from .mcts_cpp import MCTS  # type: ignore
except Exception:
    from .mcts import MCTS  # fallback to Python

__all__ = ["MCTS"]
