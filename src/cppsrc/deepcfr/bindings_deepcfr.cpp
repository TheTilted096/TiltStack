#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "CFRGame.h"
#include "CFRUtils.h"
#include "DeepCFR.h"
#include "Match.h"
#include "Orchestrator.h"
#include "Reservoir.h"
#include "TPO.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Zero-copy numpy array helpers.
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

static py::array_t<uint8_t> infoSetArray(uint8_t *data, ssize_t n,
                                         py::object base) {
    return asArray2D<uint8_t>(data, n, (ssize_t)sizeof(InfoSet), base);
}

// ---------------------------------------------------------------------------

PYBIND11_MODULE(deepcfr, m) {
    m.doc() = "DeepCFR training engine — Orchestrator / Scheduler bindings.";

    m.attr("INFOSET_BYTES") = sizeof(InfoSet);
    m.attr("NUM_ACTIONS") = NUM_ACTIONS;

    m.def("load_tables", &loadTables, py::arg("clusters_dir"),
          "Load precomputed cluster-label tables from clusters_dir.");

    // -------------------------------------------------------------------------
    // CFRGame
    // -------------------------------------------------------------------------
    py::class_<CFRGame>(m, "CFRGame")
        .def(py::init<>())

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

        .def(
            "make_move",
            [](CFRGame &g, int a) { g.makeMove(static_cast<Action>(a)); },
            py::arg("action"))

        .def(
            "make_bet",
            [](CFRGame &g, int amount_milli) { g.makeBet(amount_milli); },
            py::arg("amount_milli"))

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

        .def("get_info",
             [](const CFRGame &g) {
                 InfoSet info = const_cast<CFRGame &>(g).getInfo();
                 auto arr = py::array_t<uint8_t>(
                     {(ssize_t)1, (ssize_t)sizeof(InfoSet)});
                 std::memcpy(arr.mutable_data(), &info, sizeof(InfoSet));
                 return arr;
             })

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
    // -------------------------------------------------------------------------
    py::class_<Scheduler>(m, "Scheduler")

        // -- Inference interface ----------------------------------------------
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
        // Per-request network index (0 or 1).  Shape: (batch_size,), dtype
        // int32. Use this to partition a batch between two networks in match
        // evaluation.
        .def("net_idx_data",
             [](py::object self) {
                 auto &s = self.cast<Scheduler &>();
                 return asArray1D<int>(s.netIdxData(), s.batchSize(), self);
             })
        .def("submit_batch",
             [](Scheduler &s) {
                 py::gil_scoped_release release;
                 s.submitBatch();
             })

        // -- Advantage replay buffer ------------------------------------------
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
                 return asArray1D<float>(s.policyWeightData(), s.policySize(),
                                         self);
             })

        .def("rollout_count", &Scheduler::rolloutCount);

    // -------------------------------------------------------------------------
    // Reservoir
    // -------------------------------------------------------------------------
    py::class_<Reservoir>(m, "Reservoir")
        .def(py::init([](std::size_t capacity, int numThreads,
                         py::array_t<uint8_t, py::array::c_style> inputs,
                         py::array_t<float, py::array::c_style> targets,
                         py::object weights) {
                 float *wptr = nullptr;
                 if (!weights.is_none())
                     wptr =
                         weights.cast<py::array_t<float, py::array::c_style>>()
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
        .def("size", &Reservoir::size)
        .def("reset", &Reservoir::reset);

    // -------------------------------------------------------------------------
    // Orchestrator
    //
    // The C++ Orchestrator is a generic thread pool (startIteration takes
    // arbitrary TaskFactory / StopPredicate / WorkerSetup closures).  The
    // Python-facing convenience methods below build those closures for the two
    // supported use cases: CFR/TPO training and match evaluation.
    // -------------------------------------------------------------------------
    py::class_<Orchestrator>(m, "Orchestrator")
        .def(py::init([](int numThreads, uint64_t seed) {
                 return new Orchestrator(numThreads, seed);
             }),
             py::arg("num_threads"),
             py::arg("seed") = (uint64_t)0xdeadbeefcafe1234ULL)

        .def_readonly("schedulers", &Orchestrator::schedulerPtrs,
                      py::return_value_policy::reference_internal)

        .def("num_threads", &Orchestrator::numThreads)

        // -- DeepCFR training iteration ---------------------------------------
        // Builds a DeepCFR factory + sample-quota stop predicate and wakes all
        // workers.  pol_res is optional: pass it once the advantage network is
        // warm enough that policy samples are trustworthy (e.g. after iter 50).
        // Non-blocking.
        .def(
            "start_deepcfr_iteration",
            [](Orchestrator &o, bool hero, int t, int totalSamples,
               Reservoir &advRes0, Reservoir &advRes1, py::object polResObj) {
                Reservoir *polRes = polResObj.is_none()
                                        ? nullptr
                                        : &polResObj.cast<Reservoir &>();
                Reservoir *advRes = hero ? &advRes1 : &advRes0;
                std::size_t nAtStart =
                    advRes->nSeen.load(std::memory_order_relaxed);

                o.startIteration(
                    [hero, t](Scheduler &sched) -> Task<float> {
                        CFRGame g;
                        g.begin(STARTING_STACK, STARTING_STACK, hero);
                        return DeepCFR::rollout(std::move(g), hero, t, sched);
                    },
                    [advRes, nAtStart, totalSamples]() -> bool {
                        return advRes->nSeen.load(std::memory_order_relaxed) -
                                   nAtStart >=
                               static_cast<std::size_t>(totalSamples);
                    },
                    [advRes, polRes](Scheduler &sched) {
                        sched.advReservoir = advRes;
                        sched.polReservoir = polRes;
                        sched.syncSample = false;
                        sched.sampleMutex = nullptr;
                    });
            },
            py::arg("hero"), py::arg("t"),
            py::arg("total_samples") = 10'000'000, py::arg("adv_res0"),
            py::arg("adv_res1"), py::arg("pol_res") = py::none())

        // -- TPO training iteration -------------------------------------------
        // Builds a TPO factory + sample-quota stop predicate and wakes all
        // workers.  Collects both advantage and policy samples.  Non-blocking.
        .def(
            "start_tpo_iteration",
            [](Orchestrator &o, bool hero, int totalSamples, Reservoir &advRes0,
               Reservoir &advRes1, Reservoir &polRes) {
                Reservoir *advRes = hero ? &advRes1 : &advRes0;
                std::size_t nAtStart =
                    advRes->nSeen.load(std::memory_order_relaxed);
                auto sharedMutex = std::make_shared<std::mutex>();

                o.startIteration(
                    [hero](Scheduler &sched) -> Task<float> {
                        CFRGame g;
                        g.begin(STARTING_STACK, STARTING_STACK, hero);
                        return TPO::rollout(std::move(g), hero, sched);
                    },
                    [advRes, nAtStart, totalSamples]() -> bool {
                        return advRes->nSeen.load(std::memory_order_relaxed) -
                                   nAtStart >=
                               static_cast<std::size_t>(totalSamples);
                    },
                    [advRes, &polRes, sharedMutex](Scheduler &sched) {
                        sched.advReservoir = advRes;
                        sched.polReservoir = &polRes;
                        sched.syncSample = true;
                        sched.sampleMutex = sharedMutex;
                    });
            },
            py::arg("hero"), py::arg("total_samples") = 10'000'000,
            py::arg("adv_res0"), py::arg("adv_res1"), py::arg("pol_res"))

        // -- Match iteration --------------------------------------------------
        // Builds a Match::gamePair factory and pair-count stop predicate.
        // Per-pair payoffs are stored in each Scheduler's sbPayoffs / bbPayoffs
        // vectors; collect them after wait_iteration() via
        // collect_match_payoffs().
        //
        // flags bit layout (see Match.h):
        //   bit 0 : net0 argmax   bit 2 : net0 prune
        //   bit 1 : net1 argmax   bit 3 : net1 prune
        .def(
            "start_match",
            [](Orchestrator &o, int totalPairs, uint8_t flags) {
                auto launched = std::make_shared<std::atomic<int>>(0);

                o.startIteration(
                    [flags, launched](Scheduler &sched) -> Task<float> {
                        launched->fetch_add(1, std::memory_order_relaxed);
                        return Match::gamePair(flags, sched);
                    },
                    [launched, totalPairs]() -> bool {
                        return launched->load(std::memory_order_relaxed) >=
                               totalPairs;
                    },
                    [](Scheduler &sched) {
                        sched.advReservoir = nullptr;
                        sched.polReservoir = nullptr;
                        sched.syncSample = false;
                        sched.sampleMutex = nullptr;
                        sched.sbPayoffs.clear();
                        sched.bbPayoffs.clear();
                    });
            },
            py::arg("total_pairs"), py::arg("flags"))

        // Concatenate per-pair payoffs from all scheduler threads.
        // Returns (sb_payoffs, bb_payoffs) as numpy arrays of shape (N_pairs,).
        // Call after wait_iteration().
        .def("collect_match_payoffs",
             [](const Orchestrator &o) {
                 std::vector<float> sb, bb;
                 for (const Scheduler *s : o.schedulerPtrs) {
                     sb.insert(sb.end(), s->sbPayoffs.begin(),
                               s->sbPayoffs.end());
                     bb.insert(bb.end(), s->bbPayoffs.begin(),
                               s->bbPayoffs.end());
                 }
                 auto to_array = [](std::vector<float> &v) {
                     auto arr = py::array_t<float>(v.size());
                     std::copy(v.begin(), v.end(), arr.mutable_data());
                     return arr;
                 };
                 return py::make_tuple(to_array(sb), to_array(bb));
             })

        .def("wait_iteration",
             [](Orchestrator &o) {
                 py::gil_scoped_release release;
                 o.waitIteration();
             })

        .def(
            "pop",
            [](Orchestrator &o) -> Scheduler * {
                py::gil_scoped_release release;
                return o.pop();
            },
            py::return_value_policy::reference)

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
