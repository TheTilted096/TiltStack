#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "CFRUtils.h"
#include "Orchestrator.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Zero-copy numpy array helpers.
//
// base is a Python object whose lifetime must cover the underlying buffer —
// typically `self` (the Scheduler or Orchestrator). Passing it here causes
// numpy to increment its refcount, preventing premature garbage collection
// while the array is alive. numpy will never try to free the data pointer.
// ---------------------------------------------------------------------------

template <typename T>
static py::array_t<T> asArray1D(T *data, ssize_t n, py::object base) {
    ssize_t stride = (ssize_t)sizeof(T);
    return py::array_t<T>(std::vector<ssize_t>{n}, std::vector<ssize_t>{stride},
                          data, base);
}

template <typename T>
static py::array_t<T> asArray2D(T *data, ssize_t rows, ssize_t cols,
                                py::object base) {
    ssize_t stride = (ssize_t)sizeof(T);
    return py::array_t<T>(std::vector<ssize_t>{rows, cols},
                          std::vector<ssize_t>{cols * stride, stride}, data,
                          base);
}

// InfoSet contains mixed types (uint64 cards, float scalars, uint16 buckets,
// bools). Expose as raw uint8 bytes; Python decodes field-by-field using a
// numpy structured dtype that mirrors the C++ layout.
static py::array_t<uint8_t> infoSetArray(uint8_t *data, ssize_t n,
                                         py::object base) {
    return asArray2D<uint8_t>(data, n, (ssize_t)sizeof(InfoSet), base);
}

// ---------------------------------------------------------------------------

PYBIND11_MODULE(deepcfr, m) {
    m.doc() = "DeepCFR training engine — Orchestrator / Scheduler bindings.";

    // Expose sizeof(InfoSet) so Python can build the matching structured dtype
    // without hard-coding the number.
    m.attr("INFOSET_BYTES") = sizeof(InfoSet);
    m.attr("NUM_ACTIONS") = NUM_ACTIONS;

    // -------------------------------------------------------------------------
    // Global initialisation — call once before constructing an Orchestrator.
    // -------------------------------------------------------------------------
    m.def("load_tables", &loadTables, py::arg("clusters_dir"),
          "Load precomputed EHS and cluster-label tables from clusters_dir.");

    // -------------------------------------------------------------------------
    // Scheduler
    //
    // Python never constructs a Scheduler directly. It receives Scheduler*
    // from Orchestrator.pop() and accesses it only between pop() and the
    // matching submit_batch() call. The object is owned by the worker thread
    // and is valid for the lifetime of the Orchestrator.
    // -------------------------------------------------------------------------
    py::class_<Scheduler>(m, "Scheduler")

        // -- Inference interface ----------------------------------------------
        // Called once per pop() cycle:
        //   inputs  = sched.input_data()          # read → feed to network
        //   outputs = sched.output_data()          # write ← network results
        //   sched.submit_batch()                   # unblock worker

        .def("batch_size", &Scheduler::batchSize)

        .def("input_data",
             [](py::object self) {
                 auto &s = self.cast<Scheduler &>();
                 return infoSetArray(reinterpret_cast<uint8_t *>(s.inputData()),
                                     s.batchSize(), self);
             })
        .def("output_data",
             [](py::object self) {
                 auto &s = self.cast<Scheduler &>();
                 return asArray2D<float>(
                     reinterpret_cast<float *>(s.outputData()), s.batchSize(),
                     (ssize_t)NUM_ACTIONS, self);
             })
        .def("submit_batch",
             [](Scheduler &s) {
                 py::gil_scoped_release release;
                 s.submitBatch();
             })

        // -- Advantage replay buffer ------------------------------------------
        // Harvest after wait_iteration(); cleared at the start of the next
        // iteration automatically by the worker.

        .def("advantage_size", &Scheduler::advantageSize)

        .def("advantage_input_data",
             [](py::object self) {
                 auto &s = self.cast<Scheduler &>();
                 return infoSetArray(
                     reinterpret_cast<uint8_t *>(s.advantageInputData()),
                     s.advantageSize(), self);
             })
        .def("advantage_output_data",
             [](py::object self) {
                 auto &s = self.cast<Scheduler &>();
                 return asArray2D<float>(
                     reinterpret_cast<float *>(s.advantageOutputData()),
                     s.advantageSize(), (ssize_t)NUM_ACTIONS, self);
             })

        // -- Policy replay buffer ---------------------------------------------

        .def("policy_size", &Scheduler::policySize)

        .def("policy_input_data",
             [](py::object self) {
                 auto &s = self.cast<Scheduler &>();
                 return infoSetArray(
                     reinterpret_cast<uint8_t *>(s.policyInputData()),
                     s.policySize(), self);
             })
        .def("policy_output_data",
             [](py::object self) {
                 auto &s = self.cast<Scheduler &>();
                 return asArray2D<float>(
                     reinterpret_cast<float *>(s.policyOutputData()),
                     s.policySize(), (ssize_t)NUM_ACTIONS, self);
             })
        .def("policy_weight_data",
             [](py::object self) {
                 auto &s = self.cast<Scheduler &>();
                 return asArray1D<int>(s.policyWeightData(), s.policySize(),
                                      self);
             })

        // -- Rollout counter --------------------------------------------------
        .def("rollout_count", &Scheduler::rolloutCount);

    // -------------------------------------------------------------------------
    // Orchestrator
    // -------------------------------------------------------------------------
    py::class_<Orchestrator>(m, "Orchestrator")
        .def(py::init<int, uint64_t>(), py::arg("num_threads"),
             py::arg("seed") = (uint64_t)0xdeadbeefcafe1234ULL)

        .def_readonly("schedulers", &Orchestrator::schedulerPtrs,
                      py::return_value_policy::reference_internal)

        .def("num_threads", &Orchestrator::numThreads)

        .def("start_iteration", &Orchestrator::startIteration, py::arg("hero"),
             py::arg("t"), py::arg("total_samples") = 10'000'000)

        // Blocks on a condition variable — release the GIL so Python threads
        // (e.g. a data-loading thread) can run while we wait.
        .def("wait_iteration",
             [](Orchestrator &o) {
                 py::gil_scoped_release release;
                 o.waitIteration();
             })

        // Blocks until a worker pushes to the inference queue.
        // Returns None when a worker pushes a completion sentinel.
        .def(
            "pop",
            [](Orchestrator &o) -> Scheduler * {
                py::gil_scoped_release release;
                return o.pop();
            },
            py::return_value_policy::reference)

        .def("clear_buffers", &Orchestrator::clearBuffers)
        .def("drain_pool", &Orchestrator::drainPool);
}
