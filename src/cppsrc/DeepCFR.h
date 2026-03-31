#pragma once

#include "CFRGame.h"
#include "CFRUtils.h"
#include "Coroutine.h"
#include "Scheduler.h"

// ---------------------------------------------------------------------------
// InferenceAwaitable
// Suspends the calling coroutine, registers an inference request with the
// scheduler, and resumes once the scheduler has filled in the result.
// ---------------------------------------------------------------------------

struct InferenceAwaitable {
    InfoSet    input;
    Scheduler& sched;
    std::size_t index = 0;

    bool    await_ready()                noexcept { return false; }
    void    await_suspend(Handle handle) noexcept { index = sched.enqueueInference(input, handle); }
    Regrets await_resume()               noexcept { return sched.pendingOutputs[index]; }
};

// ---------------------------------------------------------------------------
// DeepCFR
// One instance per root rollout. Owns the CFRGame that the rollout traverses.
// rng is shared across all instances on the same thread.
// ---------------------------------------------------------------------------

class DeepCFR {
public:
    CFRGame game;

    static Action   sampleAction(const Strategy& strat, const ActionList& moves, int numMoves);
    static Strategy getInstantStrat(const Regrets& r, const ActionList& moves, int numMoves);

    Task<float> rollout(bool hero, int t, Scheduler& sched);
};
