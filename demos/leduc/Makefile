# Makefile for TiltStack

ifeq ($(OS),Windows_NT)
	SHELL := cmd.exe
	PYTHON := py
	COPY_MODULE := copy /Y build\*.pyd src\pysrc\ >nul 2>&1 || echo.
else
	PYTHON := python3
	COPY_MODULE := cp -f build/*.so src/pysrc/
endif

.PHONY: install build test clean

build:
	$(PYTHON) setup.py build_ext --build-lib=build
	$(COPY_MODULE)

install:
	$(PYTHON) -m pip install pybind11

test: build
	$(PYTHON) src/pysrc/Leduc.py

ifeq ($(OS),Windows_NT)
clean:
	-if exist build rmdir /S /Q build
	-del /Q src\pysrc\*.pyd 2>nul
	-if exist output rmdir /S /Q output
else
clean:
	rm -rf build src/pysrc/*.so output
endif
