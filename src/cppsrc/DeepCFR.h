#pragma once

#include "CFRGame.h"
#include "Coroutine.h"
#include "Scheduler.h"

// ---------------------------------------------------------------------------
// InferenceAwaitable
// Suspends the calling coroutine, registers an inference request with the
// scheduler, and resumes once the scheduler has filled in the result.
// ---------------------------------------------------------------------------

struct InferenceAwaitable {
    InfoSet input;
    Scheduler &sched;
    std::size_t index = 0;

    bool await_ready() noexcept { return false; }
    void await_suspend(Handle handle) noexcept {
        index = sched.enqueueInference(input, handle);
    }
    Regrets await_resume() noexcept { return sched.pendingOutputs[index]; }
};

// ---------------------------------------------------------------------------
// DeepCFR namespace
//
// rollout() takes CFRGame by value so the game state lives inside the
// heap-allocated coroutine frame and remains valid across all suspension
// points.  A member-function coroutine would capture `this`; calling it on
// a temporary (the natural spawn-site pattern) would leave that pointer
// dangling before the coroutine ever resumed.
// ---------------------------------------------------------------------------

namespace DeepCFR {

Action sampleAction(const Strategy &strat, const ActionList &moves,
                    int numMoves);
Strategy getInstantStrat(const Regrets &r, const ActionList &moves,
                         int numMoves);

// Root entry point: owns `game` by value in its coroutine frame, then
// immediately delegates to traverse().  Taking by value here (rather than
// as a member) keeps the frame alive across all suspension points without
// capturing a pointer to a temporary.
Task<float> rollout(CFRGame game, bool hero, int t, Scheduler &sched);

// Recursive worker: takes `game` by reference so the same CFRGame object is
// shared across the entire tree.  makeMove/unmakeMove maintain game state;
// no CFRGame copies are made, avoiding shallow-copy of hand_indexer_t.
Task<float> traverse(CFRGame &game, bool hero, int t, Scheduler &sched);

} // namespace DeepCFR
