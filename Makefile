# Makefile for TiltStack

ifeq ($(OS),Windows_NT)
	PYTHON := py
	COPY_MODULE := copy /Y build\*.pyd src\pysrc\ >nul 2>&1 || echo.
	CLEAN := if exist build rmdir /S /Q build & del /Q src\pysrc\*.pyd 2>nul || echo.
else
	PYTHON := python3
	COPY_MODULE := cp -f build/*.so src/pysrc/
	CLEAN := rm -rf build && rm -f src/pysrc/*.so
endif

.PHONY: build test clean

build:
	$(PYTHON) setup.py build_ext --build-lib=build
	$(COPY_MODULE)

test: build
	$(PYTHON) src/pysrc/test_example.py

clean:
	$(CLEAN)
