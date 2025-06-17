# MCTS-NN

This repository implements a minimal AlphaZero-style chess engine.

## Requirements

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

You will also need a C++17 compiler installed on your system.

## Building the C++ MCTS

The Monte Carlo Tree Search core is implemented in `cpp/mcts.cpp` using
pybind11. Compile the extension in place with:

```bash
python setup.py build_ext --inplace
```

Alternatively install the package in editable mode which will build the
extension automatically:

```bash
python -m pip install -e .
```

When imported, `from mcts import MCTS` will automatically use the compiled
extension.

## Running

Example scripts can be found in the `scripts/` directory. Start self-play or
arena matches with:

```bash
python scripts/selfplay_worker.py --help
python scripts/arena.py --help
```
