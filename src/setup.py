"""Setup for the hand_indexer pybind11 module."""

from setuptools import setup, Extension
import pybind11
import sys

if sys.platform == 'win32':
    extra_compile_args = ['/O2', '/openmp', '/std:c++14']
    extra_link_args = []
else:
    extra_compile_args = ['-O3', '-fopenmp', '-std=c++11']
    extra_link_args = ['-fopenmp']

setup(
    name='hand_indexer',
    ext_modules=[
        Extension(
            'hand_indexer',
            sources=[
                'cppsrc/bindings.cpp',
                'cppsrc/river_expander.cpp',
                'third_party/OMPEval/omp/HandEvaluator.cpp',
                'third_party/hand-isomorphism/hand_index.c',
                'third_party/hand-isomorphism/deck.c',
            ],
            include_dirs=[
                pybind11.get_include(),
                'third_party/hand-isomorphism',
                'third_party/OMPEval',
            ],
            language='c++',
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
)
