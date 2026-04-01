"""Setup for the deepcfr pybind11 module."""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import sys

if sys.platform == 'win32':
    cpp_flags = ['/O2', '/std:c++20']
    extra_link_args = []
else:
    cpp_flags = ['-O3', '-std=c++20']
    extra_link_args = ['-lpthread']

_CPP_ONLY_FLAGS = {'/std:c++20', '-std=c++20'}
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
    name='deepcfr',
    ext_modules=[
        Extension(
            'deepcfr',
            sources=[
                'cppsrc/bindings_deepcfr.cpp',
                'cppsrc/Orchestrator.cpp',
                'cppsrc/Scheduler.cpp',
                'cppsrc/DeepCFR.cpp',
                'cppsrc/CFRGame.cpp',
                'third_party/OMPEval/HandEvaluator.cpp',
                'third_party/hand-isomorphism/hand_index.c',
                'third_party/hand-isomorphism/deck.c',
            ],
            include_dirs=[
                pybind11.get_include(),
                'cppsrc',
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
