"""Minimal setup.py for PyBind11 module compilation."""

from setuptools import setup, Extension
import pybind11
import sys

setup(
    name='example_module',
    ext_modules=[
        Extension(
            'example_module',
            sources=['src/cppsrc/bindings.cpp'],
            include_dirs=[pybind11.get_include(), 'src/cppsrc'],
            language='c++',
            extra_compile_args=['/std:c++20'] if sys.platform == 'win32' else ['-std=c++20'],
        )
    ],
)
