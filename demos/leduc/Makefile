BUILD_DIR = build

.PHONY: build install test clean
.DEFAULT_GOAL := build

$(BUILD_DIR)/CMakeCache.txt:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

build: $(BUILD_DIR)/CMakeCache.txt
	cmake --build $(BUILD_DIR) --target leducsolver

install:
	uv sync --extra demos

test: build
	uv run python src/pysrc/Leduc.py

clean:
	rm -rf $(BUILD_DIR) src/pysrc/*.so output
