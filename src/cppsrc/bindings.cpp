/*
    pybind11 bindings for poker hand utilities.

    Exposes:
      - RiverIndexer: maps river state indices to card strings
      - RiverExpander: computes 169-dim equity vectors for river states
      - TurnIndexer:  maps turn state indices to card strings
      - TurnExpander: computes 256-dim wide-bucket histograms for turn states
*/

#include "river_expander.h"   // also pulls in OMPEval and hand_index.h (correct order)
#include "turn_expander.h"
#include "flop_expander.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// RiverIndexer — maps river state indices to human-readable card strings
// ---------------------------------------------------------------------------

static const char RANK_CHARS[] = "23456789TJQKA";
static const char SUIT_CHARS[] = "cdhs";

static std::string format_hand(const uint8_t cards[7]) {
    std::string result;
    result.reserve(30);
    for (int c = 0; c < 7; c++) {
        if (c == 2) result += " | ";
        else if (c > 0) result += " ";
        result += RANK_CHARS[cards[c] / 4];
        result += SUIT_CHARS[cards[c] % 4];
    }
    return result;
}

static std::string format_flop_hand(const uint8_t cards[5]) {
    std::string result;
    result.reserve(22);
    for (int c = 0; c < 5; c++) {
        if (c == 2) result += " | ";
        else if (c > 0) result += " ";
        result += RANK_CHARS[cards[c] / 4];
        result += SUIT_CHARS[cards[c] % 4];
    }
    return result;
}

static std::string format_turn_hand(const uint8_t cards[6]) {
    std::string result;
    result.reserve(26);
    for (int c = 0; c < 6; c++) {
        if (c == 2) result += " | ";
        else if (c > 0) result += " ";
        result += RANK_CHARS[cards[c] / 4];
        result += SUIT_CHARS[cards[c] % 4];
    }
    return result;
}

class RiverIndexer {
    hand_indexer_t indexer_;

public:
    RiverIndexer() {
        uint8_t rounds[] = {2, 3, 1, 1};
        hand_indexer_init(4, rounds, &indexer_);
    }
    ~RiverIndexer() { hand_indexer_free(&indexer_); }

    uint64_t size() const { return hand_indexer_size(&indexer_, 3); }

    std::string unindex(uint64_t index) const {
        uint8_t cards[7];
        hand_unindex(&indexer_, 3, index, cards);
        return format_hand(cards);
    }

    std::vector<std::string> batch_unindex(const std::vector<uint64_t>& indices) const {
        std::vector<std::string> result;
        result.reserve(indices.size());
        uint8_t cards[7];
        for (uint64_t idx : indices) {
            hand_unindex(&indexer_, 3, idx, cards);
            result.push_back(format_hand(cards));
        }
        return result;
    }
};

// ---------------------------------------------------------------------------
// TurnIndexer — maps turn state indices to human-readable card strings
// ---------------------------------------------------------------------------

class TurnIndexer {
    hand_indexer_t indexer_;

public:
    TurnIndexer() {
        uint8_t rounds[] = {2, 3, 1};
        hand_indexer_init(3, rounds, &indexer_);
    }
    ~TurnIndexer() { hand_indexer_free(&indexer_); }

    uint64_t size() const { return hand_indexer_size(&indexer_, 2); }

    std::string unindex(uint64_t index) const {
        uint8_t cards[6];
        hand_unindex(&indexer_, 2, index, cards);
        return format_turn_hand(cards);
    }

    std::vector<std::string> batch_unindex(const std::vector<uint64_t>& indices) const {
        std::vector<std::string> result;
        result.reserve(indices.size());
        uint8_t cards[6];
        for (uint64_t idx : indices) {
            hand_unindex(&indexer_, 2, idx, cards);
            result.push_back(format_turn_hand(cards));
        }
        return result;
    }
};

// ---------------------------------------------------------------------------
// FlopIndexer — maps flop state indices to human-readable card strings
// ---------------------------------------------------------------------------

class FlopIndexer {
    hand_indexer_t indexer_;

public:
    FlopIndexer() {
        uint8_t rounds[] = {2, 3};
        hand_indexer_init(2, rounds, &indexer_);
    }
    ~FlopIndexer() { hand_indexer_free(&indexer_); }

    uint64_t size() const { return hand_indexer_size(&indexer_, 1); }

    std::string unindex(uint64_t index) const {
        uint8_t cards[5];
        hand_unindex(&indexer_, 1, index, cards);
        return format_flop_hand(cards);
    }

    std::vector<std::string> batch_unindex(const std::vector<uint64_t>& indices) const {
        std::vector<std::string> result;
        result.reserve(indices.size());
        uint8_t cards[5];
        for (uint64_t idx : indices) {
            hand_unindex(&indexer_, 1, idx, cards);
            result.push_back(format_flop_hand(cards));
        }
        return result;
    }
};

// ---------------------------------------------------------------------------
// RiverExpander pybind11 wrappers
// ---------------------------------------------------------------------------

// Compute equity vectors for arbitrary sampled indices.
// Returns a (n, 169) uint8 numpy array.
static py::array_t<uint8_t> py_compute_sample(
    const RiverExpander& exp,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> indices)
{
    auto idx_buf = indices.request();
    const auto n = static_cast<py::ssize_t>(idx_buf.size);
    const uint64_t* idx_ptr = static_cast<const uint64_t*>(idx_buf.ptr);

    auto result = py::array_t<uint8_t>({n, (py::ssize_t)RiverExpander::DIMS});

    {
        py::gil_scoped_release release;
        exp.compute_rows(idx_ptr, n, result.mutable_data());
    }

    return result;
}

// Stream all river states in batches, calling callback(batch) for each.
// batch is a (batch_size, 169) uint8 numpy array.
// The GIL is released during each C++ compute pass and reacquired for the callback.
static void py_expand_all(const RiverExpander& exp,
                          py::object callback, int batch_size)
{
    const uint64_t total = exp.num_states();

    for (uint64_t start = 0; start < total; start += batch_size) {
        const int actual = static_cast<int>(
            std::min<uint64_t>(batch_size, total - start));

        auto batch = py::array_t<uint8_t>(
            {(py::ssize_t)actual, (py::ssize_t)RiverExpander::DIMS});

        {
            py::gil_scoped_release release;
            exp.compute_range(start, actual, batch.mutable_data());
        }

        callback(batch);
    }
}

// ---------------------------------------------------------------------------
// TurnExpander pybind11 wrappers
// ---------------------------------------------------------------------------

// Compute wide-bucket histograms for arbitrary sampled turn indices.
// Returns a (n, 256) uint8 numpy array.
static py::array_t<uint8_t> py_turn_compute_sample(
    const TurnExpander& exp,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> indices)
{
    auto idx_buf = indices.request();
    const auto n = static_cast<py::ssize_t>(idx_buf.size);
    const uint64_t* idx_ptr = static_cast<const uint64_t*>(idx_buf.ptr);

    auto result = py::array_t<uint8_t>({n, (py::ssize_t)TurnExpander::DIMS});

    {
        py::gil_scoped_release release;
        exp.compute_rows(idx_ptr, n, result.mutable_data());
    }

    return result;
}

// Stream all turn states in batches, calling callback(batch) for each.
// batch is a (batch_size, 256) uint8 numpy array.
static void py_turn_expand_all(const TurnExpander& exp,
                               py::object callback, int batch_size)
{
    const uint64_t total = exp.num_states();

    for (uint64_t start = 0; start < total; start += batch_size) {
        const int actual = static_cast<int>(
            std::min<uint64_t>(batch_size, total - start));

        auto batch = py::array_t<uint8_t>(
            {(py::ssize_t)actual, (py::ssize_t)TurnExpander::DIMS});

        {
            py::gil_scoped_release release;
            exp.compute_range(start, actual, batch.mutable_data());
        }

        callback(batch);
    }
}

// ---------------------------------------------------------------------------
// FlopExpander pybind11 wrappers
// ---------------------------------------------------------------------------

// Compute wide-bucket histograms for arbitrary sampled flop indices.
// Returns a (n, 256) uint8 numpy array.
static py::array_t<uint8_t> py_flop_compute_sample(
    const FlopExpander& exp,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> indices)
{
    auto idx_buf = indices.request();
    const auto n = static_cast<py::ssize_t>(idx_buf.size);
    const uint64_t* idx_ptr = static_cast<const uint64_t*>(idx_buf.ptr);

    auto result = py::array_t<uint8_t>({n, (py::ssize_t)FlopExpander::DIMS});

    {
        py::gil_scoped_release release;
        exp.compute_rows(idx_ptr, n, result.mutable_data());
    }

    return result;
}

// Stream all flop states in batches, calling callback(batch) for each.
// batch is a (batch_size, 256) uint8 numpy array.
static void py_flop_expand_all(const FlopExpander& exp,
                               py::object callback, int batch_size)
{
    const uint64_t total = exp.num_states();

    for (uint64_t start = 0; start < total; start += batch_size) {
        const int actual = static_cast<int>(
            std::min<uint64_t>(batch_size, total - start));

        auto batch = py::array_t<uint8_t>(
            {(py::ssize_t)actual, (py::ssize_t)FlopExpander::DIMS});

        {
            py::gil_scoped_release release;
            exp.compute_range(start, actual, batch.mutable_data());
        }

        callback(batch);
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

PYBIND11_MODULE(hand_indexer, m) {
    m.doc() = "Poker hand indexers and equity expanders for turn and river streets";

    py::class_<RiverIndexer>(m, "RiverIndexer")
        .def(py::init<>())
        .def("size", &RiverIndexer::size,
             "Total number of canonical river states (2,428,287,420)")
        .def("unindex", &RiverIndexer::unindex,
             py::arg("index"),
             "Convert a river state index to a card string")
        .def("batch_unindex", &RiverIndexer::batch_unindex,
             py::arg("indices"),
             "Convert multiple indices to card strings");

    py::class_<RiverExpander>(m, "RiverExpander")
        .def(py::init<>(),
             "Initialise card tables and hand indexers (~1s startup cost).")
        .def("num_states", &RiverExpander::num_states,
             "Total number of canonical river states (2,428,287,420)")
        .def("compute_sample", &py_compute_sample,
             py::arg("indices"),
             "Compute equity vectors for the given state indices.\n"
             "indices: 1-D uint64 array  →  returns (n, 169) uint8 array.\n"
             "Divide by 255 to obtain float equity values.")
        .def("expand_all", &py_expand_all,
             py::arg("callback"),
             py::arg("batch_size") = 1000000,
             "Stream all river states in batches.\n"
             "Calls callback(batch) for each batch where batch is (B, 169) uint8.\n"
             "GIL is released during C++ computation and reacquired for each callback.");

    py::class_<TurnIndexer>(m, "TurnIndexer")
        .def(py::init<>())
        .def("size", &TurnIndexer::size,
             "Total number of canonical turn states (55,190,538)")
        .def("unindex", &TurnIndexer::unindex,
             py::arg("index"),
             "Convert a turn state index to a card string")
        .def("batch_unindex", &TurnIndexer::batch_unindex,
             py::arg("indices"),
             "Convert multiple indices to card strings");

    py::class_<TurnExpander>(m, "TurnExpander")
        .def(py::init<std::string>(),
             py::arg("river_labels_path"),
             "Load river_labels.bin into RAM and initialise turn/river hand indexers.\n"
             "river_labels_path: path to the river_labels.bin produced by the river pipeline.")
        .def("num_states", &TurnExpander::num_states,
             "Total number of canonical turn states.")
        .def("compute_sample", &py_turn_compute_sample,
             py::arg("indices"),
             "Compute wide-bucket histograms for the given turn state indices.\n"
             "indices: 1-D uint64 array  →  returns (n, 256) uint8 array.\n"
             "Each row is a count histogram over 256 wide buckets summing to 46.\n"
             "Divide by 46.0 to obtain float32 probability vectors.")
        .def("expand_all", &py_turn_expand_all,
             py::arg("callback"),
             py::arg("batch_size") = 500000,
             "Stream all turn states in batches.\n"
             "Calls callback(batch) for each batch where batch is (B, 256) uint8.\n"
             "GIL is released during C++ computation and reacquired for each callback.");

    py::class_<FlopIndexer>(m, "FlopIndexer")
        .def(py::init<>())
        .def("size", &FlopIndexer::size,
             "Total number of canonical flop states (1,286,792)")
        .def("unindex", &FlopIndexer::unindex,
             py::arg("index"),
             "Convert a flop state index to a card string")
        .def("batch_unindex", &FlopIndexer::batch_unindex,
             py::arg("indices"),
             "Convert multiple indices to card strings");

    py::class_<FlopExpander>(m, "FlopExpander")
        .def(py::init<std::string>(),
             py::arg("turn_labels_path"),
             "Load turn_labels.bin into RAM and initialise flop/turn hand indexers.\n"
             "turn_labels_path: path to the turn_labels.bin produced by the turn pipeline.")
        .def("num_states", &FlopExpander::num_states,
             "Total number of canonical flop states.")
        .def("compute_sample", &py_flop_compute_sample,
             py::arg("indices"),
             "Compute wide-bucket histograms for the given flop state indices.\n"
             "indices: 1-D uint64 array  →  returns (n, 256) uint8 array.\n"
             "Each row is a count histogram over 256 wide buckets summing to 47.\n"
             "Divide by 47.0 to obtain float32 probability vectors.")
        .def("expand_all", &py_flop_expand_all,
             py::arg("callback"),
             py::arg("batch_size") = 500000,
             "Stream all flop states in batches.\n"
             "Calls callback(batch) for each batch where batch is (B, 256) uint8.\n"
             "GIL is released during C++ computation and reacquired for each callback.");
}
