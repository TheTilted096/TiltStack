/*
    pybind11 bindings for the hand-isomorphism river indexer.

    Exposes a RiverIndexer class that converts river state indices
    (0 to 2,428,287,419) into human-readable card strings like
    "Ah Kd | Qs Jc 7h 2s 9d".

    Built as a shared library (hand_indexer.pyd / .so) importable from Python.
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <vector>

extern "C" {
#include "hand_index.h"
}

namespace py = pybind11;

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

    ~RiverIndexer() {
        hand_indexer_free(&indexer_);
    }

    uint64_t size() const {
        return hand_indexer_size(&indexer_, 3);
    }

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

PYBIND11_MODULE(hand_indexer, m) {
    m.doc() = "River hand indexer -- maps indices to canonical 7-card hands";

    py::class_<RiverIndexer>(m, "RiverIndexer")
        .def(py::init<>())
        .def("size", &RiverIndexer::size,
             "Number of canonical river states (2,428,287,420)")
        .def("unindex", &RiverIndexer::unindex,
             "Convert a river state index to a card string",
             py::arg("index"))
        .def("batch_unindex", &RiverIndexer::batch_unindex,
             "Convert multiple indices to card strings",
             py::arg("indices"));
}
