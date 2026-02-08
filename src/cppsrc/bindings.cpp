// bindings.cpp - Pybind11 bindings for the example module

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "example.cpp"

namespace py = pybind11;

PYBIND11_MODULE(example_module, m) {
    m.doc() = "Example pybind11 module demonstrating C++ to Python bindings";

    // Bind simple functions
    m.def("add", &add, "Add two integers",
          py::arg("a"), py::arg("b"));

    m.def("multiply", &multiply, "Multiply two doubles",
          py::arg("a"), py::arg("b"));

    m.def("greet", &greet, "Return a greeting message",
          py::arg("name"));

    // Bind the Calculator class
    py::class_<Calculator>(m, "Calculator")
        .def(py::init<>(), "Create a calculator with initial value of 0")
        .def(py::init<double>(), "Create a calculator with an initial value",
             py::arg("initial_value"))
        .def("add", &Calculator::add, "Add a value to the calculator",
             py::arg("x"))
        .def("multiply", &Calculator::multiply, "Multiply the calculator value",
             py::arg("x"))
        .def("get_value", &Calculator::get_value, "Get the current value")
        .def("reset", &Calculator::reset, "Reset the calculator to 0");
}
