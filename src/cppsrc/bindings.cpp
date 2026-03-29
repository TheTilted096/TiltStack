/*
    pybind11 bindings for poker hand utilities.

    Exposes:
      - PreflopIndexer: canonical index for 2-card hole hands
      - RiverIndexer:   maps river state indices to card strings
      - RiverExpander:  computes 169-dim equity vectors + EHS + multiplicities for river states
      - TurnIndexer:    maps turn state indices to card strings
      - TurnExpander:   computes 256-dim histograms, per-state EHS, and multiplicities
      - FlopIndexer:    maps flop state indices to card strings; forward-indexes cards
      - FlopExpander:   computes 256-dim histograms, per-state EHS, and multiplicities
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

static std::string format_preflop_hand(const uint8_t cards[2]) {
    std::string result;
    result.reserve(5);
    result += RANK_CHARS[cards[0] / 4];
    result += SUIT_CHARS[cards[0] % 4];
    result += ' ';
    result += RANK_CHARS[cards[1] / 4];
    result += SUIT_CHARS[cards[1] % 4];
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
// PreflopIndexer — canonical index for 2-card hole hands (169 classes)
// ---------------------------------------------------------------------------

class PreflopIndexer {
    hand_indexer_t indexer_;

public:
    PreflopIndexer() {
        uint8_t rounds[] = {2};
        hand_indexer_init(1, rounds, &indexer_);
    }
    ~PreflopIndexer() { hand_indexer_free(&indexer_); }

    uint64_t size() const { return hand_indexer_size(&indexer_, 0); }

    uint64_t index(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> cards) const {
        auto buf = cards.request();
        if (buf.size != 2)
            throw std::invalid_argument("PreflopIndexer.index: expected array of 2 cards");
        return hand_index_last(&indexer_, static_cast<uint8_t*>(buf.ptr));
    }

    std::string unindex(uint64_t idx) const {
        uint8_t cards[2];
        hand_unindex(&indexer_, 0, idx, cards);
        return format_preflop_hand(cards);
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

    uint64_t index(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> cards) const {
        auto buf = cards.request();
        if (buf.size != 5)
            throw std::invalid_argument("FlopIndexer.index: expected array of 5 cards");
        return hand_index_last(&indexer_, static_cast<uint8_t*>(buf.ptr));
    }

    py::array_t<uint64_t> batch_index(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> cards) const {
        auto buf = cards.request();
        if (buf.ndim != 2 || buf.shape[1] != 5)
            throw std::invalid_argument("FlopIndexer.batch_index: expected (N, 5) uint8 array");
        const auto n = buf.shape[0];
        const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);

        auto result = py::array_t<uint64_t>({n});
        uint64_t* out = result.mutable_data();
        for (py::ssize_t i = 0; i < n; i++)
            out[i] = hand_index_last(&indexer_, ptr + i * 5);
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

// Stream all river states in batches, calling callback(equity, ehs, mult) for each.
// equity is (B, 169) uint8; ehs is (B,) float32; mult is (B,) uint8.
static void py_expand_all_with_ehs_mult(const RiverExpander& exp,
                                        py::object callback, int batch_size)
{
    const uint64_t total = exp.num_states();

    for (uint64_t start = 0; start < total; start += batch_size) {
        const int actual = static_cast<int>(
            std::min<uint64_t>(batch_size, total - start));

        auto equity = py::array_t<uint8_t>(
            {(py::ssize_t)actual, (py::ssize_t)RiverExpander::DIMS});
        auto ehs  = py::array_t<float>({(py::ssize_t)actual});
        auto mult = py::array_t<uint8_t>({(py::ssize_t)actual});

        {
            py::gil_scoped_release release;
            exp.compute_range_ehs_mult(start, actual,
                                       equity.mutable_data(),
                                       ehs.mutable_data(),
                                       mult.mutable_data());
        }

        callback(equity, ehs, mult);
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

// Stream all turn states, calling callback(hist, ehs, mult) for each.
// hist is (B, 256) uint8; ehs is (B,) float32; mult is (B,) uint8.
static void py_turn_expand_all_with_ehs_mult(const TurnExpander& exp,
                                             py::object callback, int batch_size)
{
    const uint64_t total = exp.num_states();

    for (uint64_t start = 0; start < total; start += batch_size) {
        const int actual = static_cast<int>(
            std::min<uint64_t>(batch_size, total - start));

        auto hist = py::array_t<uint8_t>(
            {(py::ssize_t)actual, (py::ssize_t)TurnExpander::DIMS});
        auto ehs  = py::array_t<float>({(py::ssize_t)actual});
        auto mult = py::array_t<uint8_t>({(py::ssize_t)actual});

        {
            py::gil_scoped_release release;
            exp.compute_range_ehs_mult(start, actual,
                                       hist.mutable_data(),
                                       ehs.mutable_data(),
                                       mult.mutable_data());
        }

        callback(hist, ehs, mult);
    }
}

// ---------------------------------------------------------------------------
// FlopExpander pybind11 wrappers
// ---------------------------------------------------------------------------

// Compute wide-bucket histograms and EHS and multiplicities for arbitrary
// sampled flop indices in a single pass.
// Returns (hist, ehs, mult) as a tuple:
//   hist: (n, 256) uint8
//   ehs:  (n,) float32
//   mult: (n,) uint8
static py::tuple py_flop_compute_sample_ehs_mult(
    const FlopExpander& exp,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> indices)
{
    auto idx_buf = indices.request();
    const auto n = static_cast<py::ssize_t>(idx_buf.size);
    const uint64_t* idx_ptr = static_cast<const uint64_t*>(idx_buf.ptr);

    auto hist = py::array_t<uint8_t>({n, (py::ssize_t)FlopExpander::DIMS});
    auto ehs  = py::array_t<float>({n});
    auto mult = py::array_t<uint8_t>({n});

    {
        py::gil_scoped_release release;
        exp.compute_rows_ehs_mult(idx_ptr, n,
                                  hist.mutable_data(),
                                  ehs.mutable_data(),
                                  mult.mutable_data());
    }

    return py::make_tuple(hist, ehs, mult);
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

PYBIND11_MODULE(hand_indexer, m) {
    m.doc() = "Poker hand indexers and equity expanders";

    py::class_<PreflopIndexer>(m, "PreflopIndexer")
        .def(py::init<>())
        .def("size", &PreflopIndexer::size,
             "Total number of canonical preflop hole-hand classes (169)")
        .def("index", &PreflopIndexer::index,
             py::arg("cards"),
             "Map a 2-element uint8 array of card codes to a canonical preflop index.")
        .def("unindex", &PreflopIndexer::unindex,
             py::arg("index"),
             "Convert a canonical preflop index to a card string.");

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
        .def("expand_all_with_ehs_mult", &py_expand_all_with_ehs_mult,
             py::arg("callback"),
             py::arg("batch_size") = 1000000,
             "Stream all river states in batches, yielding equity, EHS, and multiplicity.\n"
             "Calls callback(equity, ehs, mult) where equity is (B, 169) uint8,\n"
             "ehs is (B,) float32, and mult is (B,) uint8.\n"
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
        .def(py::init<std::string, std::string>(),
             py::arg("river_labels_path"),
             py::arg("river_ehs_fine_path"),
             "Load both river files into RAM and initialise turn/river hand indexers.\n"
             "river_labels_path:   path to river_labels.bin   (~4.9 GB)\n"
             "river_ehs_fine_path: path to river_ehs_fine.bin (~4.9 GB)")
        .def("num_states", &TurnExpander::num_states,
             "Total number of canonical turn states.")
        .def("compute_sample", &py_turn_compute_sample,
             py::arg("indices"),
             "Compute wide-bucket histograms for the given turn state indices.\n"
             "indices: 1-D uint64 array  →  returns (n, 256) uint8 array.\n"
             "Each row is a count histogram over 256 wide buckets summing to 46.")
        .def("expand_all_with_ehs_mult", &py_turn_expand_all_with_ehs_mult,
             py::arg("callback"),
             py::arg("batch_size") = 500000,
             "Stream all turn states in batches, yielding histograms, EHS, and multiplicities.\n"
             "Calls callback(hist, ehs, mult) where hist is (B, 256) uint8,\n"
             "ehs is (B,) float32, and mult is (B,) uint8.\n"
             "GIL is released during C++ computation and reacquired for each callback.");

    py::class_<FlopIndexer>(m, "FlopIndexer")
        .def(py::init<>())
        .def("size", &FlopIndexer::size,
             "Total number of canonical flop states (1,286,792)")
        .def("index", &FlopIndexer::index,
             py::arg("cards"),
             "Map a 5-element uint8 array of card codes to a canonical flop index.")
        .def("unindex", &FlopIndexer::unindex,
             py::arg("index"),
             "Convert a flop state index to a card string")
        .def("batch_unindex", &FlopIndexer::batch_unindex,
             py::arg("indices"),
             "Convert multiple indices to card strings")
        .def("batch_index", &FlopIndexer::batch_index,
             py::arg("cards"),
             "Map an (N, 5) uint8 array of card-code rows to a (N,) uint64 array of canonical flop indices.");

    py::class_<FlopExpander>(m, "FlopExpander")
        .def(py::init<std::string, std::string>(),
             py::arg("turn_labels_path"),
             py::arg("turn_ehs_fine_path"),
             "Load both turn files into RAM and initialise flop/turn hand indexers.\n"
             "turn_labels_path:   path to turn_labels.bin   (~110 MB)\n"
             "turn_ehs_fine_path: path to turn_ehs_fine.bin (~110 MB)")
        .def("num_states", &FlopExpander::num_states,
             "Total number of canonical flop states.")
        .def("compute_sample_ehs_mult", &py_flop_compute_sample_ehs_mult,
             py::arg("indices"),
             "Compute histograms, EHS, and multiplicities for the given flop state indices.\n"
             "indices: 1-D uint64 array  →  returns (hist, ehs, mult) tuple where\n"
             "hist is (n, 256) uint8, ehs is (n,) float32, mult is (n,) uint8.");
}
