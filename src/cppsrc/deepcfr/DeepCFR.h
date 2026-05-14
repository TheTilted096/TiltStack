#pragma once

#include "CFRGame.h"
#include "Scheduler.h"

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

// Root entry point: owns `game` by value in its coroutine frame, then
// immediately delegates to traverse().
Task<float> rollout(CFRGame game, bool hero, int t, Scheduler &sched);

// Recursive worker: takes `game` by reference so the same CFRGame object is
// shared across the entire tree.  makeMove/unmakeMove maintain game state;
// no CFRGame copies are made, avoiding shallow-copy of hand_indexer_t.
Task<float> traverse(CFRGame &game, bool hero, int t, Scheduler &sched);

} // namespace DeepCFR
