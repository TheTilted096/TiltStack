#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "CFRGame.h"
#include "CFRUtils.h"
#include "Orchestrator.h"
#include "Reservoir.h"

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
          "Load precomputed cluster-label tables from clusters_dir.");

    // -------------------------------------------------------------------------
    // CFRGame
    //
    // Lightweight game instance for single-hand feature extraction.
    // Typical evaluation usage:
    //   game = deepcfr.CFRGame()
    //   game.begin_with_cards(ss1, ss2, hero, cards9)  # cards9: list of 9 ints
    //   game.make_move(action)                          # action: int 0-4
    //   raw = game.get_info()                           # shape (1, 160) uint8
    //   actions = game.generate_actions()               # list[int]
    // -------------------------------------------------------------------------
    py::class_<CFRGame>(m, "CFRGame")
        .def(py::init<>())

        // begin_with_cards(ss1, ss2, hero, cards) where cards is a list of 9
        // card indices in [0,51], layout: [p0h0, p0h1, p1h0, p1h1, f0..f2, t,
        // r]. Unreached-street slots (turn/river when on flop) should be filled
        // with any unused card indices so hand_index_all can run cleanly;
        // their bucket values are gated by currentRound in get_info().
        .def(
            "begin_with_cards",
            [](CFRGame &g, int ss1, int ss2, bool hero,
               const std::vector<int> &cards) {
                if (cards.size() != 9)
                    throw std::invalid_argument(
                        "cards must have exactly 9 elements");
                Card c[9];
                for (int i = 0; i < 9; i++)
                    c[i] = static_cast<Card>(cards[i]);
                g.beginWithCards(ss1, ss2, hero, c);
            },
            py::arg("ss1"), py::arg("ss2"), py::arg("hero"), py::arg("cards"))

        // make_move(action_int): action is an int matching the Action enum
        // (CHECK=0, CALL=1, BET50=2, BET100=3, ALLIN=4).
        .def(
            "make_move",
            [](CFRGame &g, int a) { g.makeMove(static_cast<Action>(a)); },
            py::arg("action"))

        // make_bet(amount_milli): apply a continuous bet given as a milli-chip
        // amount.  Records the nearest pot-fraction abstract action label
        // (CHECK/CALL/BET50/BET100/ALLIN) while updating financial state with
        // the exact amount provided.
        .def(
            "make_bet",
            [](CFRGame &g, int amount_milli) { g.makeBet(amount_milli); },
            py::arg("amount_milli"))

        // generate_actions() -> list[int]: legal abstract action indices.
        .def("generate_actions",
             [](CFRGame &g) {
                 ActionList alist;
                 int n = g.generateActions(alist, /*prune=*/true);
                 std::vector<int> out;
                 out.reserve(n);
                 for (int i = 0; i < n; i++)
                     out.push_back(static_cast<int>(alist[i]));
                 return out;
             })

        // get_info() -> np.ndarray shape (1, INFOSET_BYTES) dtype uint8.
        // The returned array owns a copy of the InfoSet struct so the CFRGame
        // object can be mutated freely after this call.
        .def("get_info",
             [](const CFRGame &g) {
                 InfoSet info = const_cast<CFRGame &>(g).getInfo();
                 auto arr = py::array_t<uint8_t>(
                     {(ssize_t)1, (ssize_t)sizeof(InfoSet)});
                 std::memcpy(arr.mutable_data(), &info, sizeof(InfoSet));
                 return arr;
             })

        // Read-only state accessors used by the agent for action mapping.
        .def_property_readonly(
            "pot", [](const CFRGame &g) { return g.history[g.ply].pot; })
        .def_property_readonly(
            "to_call", [](const CFRGame &g) { return g.history[g.ply].toCall; })
        .def_property_readonly("stm",
                               [](const CFRGame &g) {
                                   return static_cast<int>(
                                       g.history[g.ply].stm);
                               })
        .def_property_readonly(
            "current_round",
            [](const CFRGame &g) { return static_cast<int>(g.currentRound); })
        .def_property_readonly("stacks",
                               [](const CFRGame &g) {
                                   return std::vector<int>{g.stacks[0],
                                                           g.stacks[1]};
                               })
        .def_property_readonly(
            "is_terminal", [](const CFRGame &g) { return g.isTerminal != 0; });

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
    // Reservoir
    //
    // Python owns the numpy arrays (for pinned-memory / zero-copy PyTorch
    // access); the C++ Reservoir writes into them directly from worker threads
    // during flushBatch(), overlapping reservoir writes with GPU inference.
    //
    // The numpy arrays must remain valid for the lifetime of the Reservoir and
    // any Orchestrator that holds a pointer to it.
    // -------------------------------------------------------------------------
    py::class_<Reservoir>(m, "Reservoir")
        .def(py::init([](std::size_t capacity, int numThreads,
                         py::array_t<uint8_t, py::array::c_style> inputs,
                         py::array_t<float, py::array::c_style> targets,
                         py::object weights) {
                 int32_t *wptr = nullptr;
                 if (!weights.is_none())
                     wptr =
                         weights
                             .cast<py::array_t<int32_t, py::array::c_style>>()
                             .mutable_data();
                 return new Reservoir(capacity, numThreads,
                                      inputs.mutable_data(),
                                      targets.mutable_data(), wptr);
             }),
             py::arg("capacity"), py::arg("num_threads"), py::arg("inputs"),
             py::arg("targets"), py::arg("weights") = py::none())
        .def_property_readonly("n_seen",
                               [](const Reservoir &r) {
                                   return r.nSeen.load(
                                       std::memory_order_relaxed);
                               })
        .def("size", &Reservoir::size);

    // -------------------------------------------------------------------------
    // Orchestrator
    // -------------------------------------------------------------------------
    py::class_<Orchestrator>(m, "Orchestrator")
        .def(py::init([](int numThreads, Reservoir &advRes0, Reservoir &advRes1,
                         py::object polResObj, uint64_t seed) {
                 Reservoir *polPtr = polResObj.is_none()
                                         ? nullptr
                                         : &polResObj.cast<Reservoir &>();
                 return new Orchestrator(numThreads, &advRes0, &advRes1, polPtr,
                                         seed);
             }),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
             py::arg("num_threads"), py::arg("adv_res0"), py::arg("adv_res1"),
             py::arg("pol_res") = py::none(),
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

        // Non-blocking drain — returns a list of all Schedulers already
        // waiting in the queue. None entries are sentinels (worker finished).
        // Call immediately after pop() to coalesce ready batches into one
        // GPU forward pass.
        .def("drain",
             [](Orchestrator &o) -> py::list {
                 py::list result;
                 for (Scheduler *s : o.drain()) {
                     if (s == nullptr)
                         result.append(py::none());
                     else
                         result.append(
                             py::cast(s, py::return_value_policy::reference));
                 }
                 return result;
             })

        .def("clear_buffers", &Orchestrator::clearBuffers)
        .def("drain_pool", &Orchestrator::drainPool)

        // Returns a list of dicts, one per worker thread, with pool stats from
        // the last iteration. Call after wait_iteration().
        .def("get_pool_stats",
             [](const Orchestrator &o) {
                 py::list result;
                 for (const auto &s : o.getPoolStats()) {
                     py::dict sizes;
                     for (const auto &[sz, cnt] : s.allSizes)
                         sizes[py::int_(sz)] = cnt;
                     result.append(sizes);
                 }
                 return result;
             })
        .def("clear_pool_stats", &Orchestrator::clearPoolStats);
}
