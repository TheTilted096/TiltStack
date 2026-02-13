#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Node.cpp"
#include "Leduc.cpp"
#include "BestResponse.cpp"

namespace py = pybind11;

PYBIND11_MODULE(leducsolver, m) {
    m.doc() = "Leduc Poker CFR solver";

    py::enum_<Action>(m, "Action")
        .value("CHECK", Check)
        .value("BET", Bet)
        .value("RAISE", Raise);

    py::enum_<Rank>(m, "Rank")
        .value("JACK", Jack)
        .value("QUEEN", Queen)
        .value("KING", King);

    py::class_<ActionList>(m, "ActionList")
        .def_readonly("len", &ActionList::len)
        .def("__getitem__", [](const ActionList& al, int i){ return al.al[i]; })
        .def("__len__", [](const ActionList& al){ return al.len; })
        .def("__iter__", [](const ActionList& al){
            return py::make_iterator(al.al.begin(), al.al.begin() + al.len);
        }, py::keep_alive<0, 1>());

    py::class_<NodeInfo>(m, "NodeInfo")
        .def(py::init<const Hash&>())
        .def_readonly("hash", &NodeInfo::hash)
        .def_readonly("private_card", &NodeInfo::private_card)
        .def_readonly("shared_card", &NodeInfo::shared_card)
        .def_readonly("bet_round", &NodeInfo::bet_round)
        .def_readonly("raises", &NodeInfo::raises)
        .def("stm", &NodeInfo::stm)
        .def("moves", &NodeInfo::moves);

    py::class_<Node>(m, "Node")
        .def("get_stored_strategy", &Node::get_stored_strategy);

    py::class_<LeducSolver>(m, "LeducSolver")
        .def(py::init<>())
        .def("cfr", &LeducSolver::cfr)
        .def("__getitem__", [](LeducSolver& s, int i) -> Node& { return s.nodes[i]; },
             py::return_value_policy::reference_internal);

    py::class_<BestResponse>(m, "BestResponse")
        .def(py::init<>())
        .def("load_strategy", &BestResponse::load_strategy)
        .def("compute", &BestResponse::compute)
        .def("get_ev", &BestResponse::get_ev)
        .def("get_br_strategy_at", &BestResponse::get_br_strategy_at)
        .def("get_full_br_strategy", &BestResponse::get_full_br_strategy);
}
