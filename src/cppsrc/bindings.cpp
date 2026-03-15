/*
    pybind11 bindings for poker hand utilities.

    Exposes:
      - RiverIndexer: maps river state indices to card strings
      - RiverExpander: computes 169-dim equity vectors for river states
*/

#include "river_expander.h"   // also pulls in OMPEval and hand_index.h (correct order)

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
// Module registration
// ---------------------------------------------------------------------------

PYBIND11_MODULE(hand_indexer, m) {
    m.doc() = "River hand indexer and equity expander";

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
}
