from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from torch.utils.cpp_extension import include_paths, library_paths

ext_modules = [
    Pybind11Extension(
        'mcts.mcts_cpp',
        ['cpp/mcts.cpp'],
        include_dirs=include_paths(),
        library_dirs=library_paths(),
        libraries=['torch', 'c10'],
        cxx_std=17,
    ),
]

setup(
    name='mcts_cpp',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
