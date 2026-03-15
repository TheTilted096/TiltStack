"""Setup for the hand_indexer pybind11 module."""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import sys

if sys.platform == 'win32':
    cpp_flags = ['/O2', '/openmp', '/std:c++14']
    extra_link_args = []
else:
    cpp_flags = ['-O3', '-fopenmp', '-std=c++11']
    extra_link_args = ['-fopenmp']

# Flags that are C++-only and must not be forwarded to the C compiler.
_CPP_ONLY_FLAGS = {'/std:c++14', '-std=c++11', '-std=c++14', '-std=c++17'}
_c_flags = [f for f in cpp_flags if f not in _CPP_ONLY_FLAGS]


class BuildExtMixed(build_ext):
    """Strip C++-standard flags when compiling plain C source files."""

    def build_extension(self, ext):
        original = self.compiler._compile

        def _compile(obj, src, ext_, cc_args, extra_postargs, pp_opts):
            if src.endswith('.c'):
                extra_postargs = _c_flags
            return original(obj, src, ext_, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = _compile
        try:
            super().build_extension(ext)
        finally:
            self.compiler._compile = original


setup(
    name='hand_indexer',
    ext_modules=[
        Extension(
            'hand_indexer',
            sources=[
                'cppsrc/bindings.cpp',
                'cppsrc/river_expander.cpp',
                'cppsrc/turn_expander.cpp',
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
            extra_compile_args=cpp_flags,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={'build_ext': BuildExtMixed},
)
