cmake_minimum_required(VERSION 3.20)
project(tiltstack CXX C)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

include(FetchContent)

FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)

FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)

# hand-isomorphism and OMPEval have no CMakeLists.txt — populate only.
FetchContent_Declare(hand_isomorphism
    GIT_REPOSITORY https://github.com/kdub0/hand-isomorphism.git
    GIT_TAG        master
)
FetchContent_Declare(OMPEval
    GIT_REPOSITORY https://github.com/zekyll/OMPEval.git
    GIT_TAG        master
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(pybind11 googletest)

FetchContent_GetProperties(hand_isomorphism)
if(NOT hand_isomorphism_POPULATED)
    FetchContent_Populate(hand_isomorphism)
endif()

FetchContent_GetProperties(OMPEval)
if(NOT ompeval_POPULATED)
    FetchContent_Populate(OMPEval)
endif()

# ---------------------------------------------------------------------------
# Shared C/C++ libraries from vendored sources
# ---------------------------------------------------------------------------

add_library(hand_iso STATIC
    ${hand_isomorphism_SOURCE_DIR}/src/hand_index.c
    ${hand_isomorphism_SOURCE_DIR}/src/deck.c
)
target_include_directories(hand_iso PUBLIC ${hand_isomorphism_SOURCE_DIR}/src)
target_compile_options(hand_iso PRIVATE -O3)
target_link_libraries(hand_iso PUBLIC OpenMP::OpenMP_C)

add_library(ompeval STATIC
    ${ompeval_SOURCE_DIR}/omp/HandEvaluator.cpp
)
target_include_directories(ompeval PUBLIC ${ompeval_SOURCE_DIR}/omp)
set_target_properties(ompeval PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)
target_compile_options(ompeval PRIVATE -O3)

# ---------------------------------------------------------------------------
# hand_indexer pybind11 extension
# ---------------------------------------------------------------------------

pybind11_add_module(hand_indexer
    cppsrc/clustering/bindings.cpp
    cppsrc/clustering/river_expander.cpp
    cppsrc/clustering/turn_expander.cpp
    cppsrc/clustering/flop_expander.cpp
)
set_target_properties(hand_indexer PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/pysrc/clustering"
)
target_compile_options(hand_indexer PRIVATE -O3)
target_link_libraries(hand_indexer PRIVATE hand_iso ompeval OpenMP::OpenMP_CXX)
target_include_directories(hand_indexer PRIVATE cppsrc/clustering)

# ---------------------------------------------------------------------------
# deepcfr pybind11 extension
# ---------------------------------------------------------------------------

pybind11_add_module(deepcfr
    cppsrc/deepcfr/bindings_deepcfr.cpp
    cppsrc/deepcfr/Reservoir.cpp
    cppsrc/deepcfr/Orchestrator.cpp
    cppsrc/deepcfr/Scheduler.cpp
    cppsrc/deepcfr/DeepCFR.cpp
    cppsrc/deepcfr/CFRGame.cpp
)
set_target_properties(deepcfr PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/pysrc/deepcfr"
)
target_compile_options(deepcfr PRIVATE -O3)
target_link_libraries(deepcfr PRIVATE hand_iso ompeval Threads::Threads)
target_include_directories(deepcfr PRIVATE cppsrc/deepcfr)

# ---------------------------------------------------------------------------
# C++ unit tests
# ---------------------------------------------------------------------------

add_executable(cfr_tests
    cppsrc/test/test_cfrgame.cpp
    cppsrc/test/test_deepcfr.cpp
    cppsrc/test/test_rng.cpp
    cppsrc/deepcfr/CFRGame.cpp
    cppsrc/deepcfr/DeepCFR.cpp
    cppsrc/deepcfr/Scheduler.cpp
)
set_target_properties(cfr_tests PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
target_compile_options(cfr_tests PRIVATE -O3)
target_link_libraries(cfr_tests PRIVATE hand_iso ompeval GTest::gtest_main Threads::Threads)
target_include_directories(cfr_tests PRIVATE cppsrc/deepcfr)

include(GoogleTest)
enable_testing()
gtest_discover_tests(cfr_tests)
