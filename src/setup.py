"""Minimal setup.py for PyBind11 hand_indexer module."""

from setuptools import setup, Extension
import pybind11
import sys

setup(
    name='hand_indexer',
    ext_modules=[
        Extension(
            'hand_indexer',
            sources=[
                'cppsrc/bindings.cpp',
                'third_party/hand-isomorphism/hand_index.c',
                'third_party/hand-isomorphism/deck.c',
            ],
            include_dirs=[
                pybind11.get_include(),
                'third_party/hand-isomorphism',
            ],
            language='c++',
            extra_compile_args=['/O2'] if sys.platform == 'win32' else ['-O3'],
        )
    ],
)
