# MCTS-NN

This repository implements a minimal AlphaZero-style chess engine.

## Building the C++ MCTS

A new pybind11 extension provides a faster Monte Carlo Tree Search.
Build the extension with:

```bash
python setup.py build_ext --inplace
```

Ensure `pybind11`, PyTorch and a C++17 compiler are available.
You can also install the package in editable mode which will compile the
extension automatically:

```bash
python -m pip install -e .
```
