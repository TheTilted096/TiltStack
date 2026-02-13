# Makefile for TiltStack

ifeq ($(OS),Windows_NT)
	SHELL := cmd.exe
	PYTHON := py
	COPY_MODULE := copy /Y build\*.pyd src\pysrc\ >nul 2>&1 || echo.
	CLEAN := if exist build rmdir /S /Q build & del /Q src\pysrc\*.pyd 2>nul || echo. & del /Q leduc_results.txt leduc_strategy.csv br_results.txt br_p0_results.txt br_p1_results.txt br_p0_strategy.csv br_p1_strategy.csv 2>nul || echo.
else
	PYTHON := python3
	COPY_MODULE := cp -f build/*.so src/pysrc/
	CLEAN := rm -rf build && rm -f src/pysrc/*.so && rm -f leduc_results.txt leduc_strategy.csv br_results.txt br_p0_results.txt br_p1_results.txt br_p0_strategy.csv br_p1_strategy.csv
endif

.PHONY: install build test best clean

build:
	$(PYTHON) setup.py build_ext --build-lib=build
	$(COPY_MODULE)

install:
	$(PYTHON) -m pip install pybind11

test: build
	$(PYTHON) src/pysrc/Leduc.py

best: build
	$(PYTHON) src/pysrc/BestResponse.py

clean:
	$(CLEAN)
