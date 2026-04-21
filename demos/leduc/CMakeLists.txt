cmake_minimum_required(VERSION 3.20)
project(leducsolver CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

# Tell pybind11 to use CMake's find_package(Python) instead of its own legacy
# FindPythonInterp/FindPythonLibs, so it picks up the same Python we found.
set(PYBIND11_FINDPYTHON ON)

include(FetchContent)

FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(leducsolver src/cppsrc/bindings.cpp)
set_target_properties(leducsolver PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/pysrc"
)
target_compile_options(leducsolver PRIVATE -O3)
target_include_directories(leducsolver PRIVATE src/cppsrc)
