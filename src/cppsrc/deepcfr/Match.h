#pragma once

#include "CFRGame.h"
#include "Coroutine.h"
#include "Scheduler.h"

// ---------------------------------------------------------------------------
// Match evaluation coroutines.
//
// singleGame plays one pass of a hand.  At each decision node it suspends via
// InferenceAwaitable (routing to network netIdx = stm ^ passB) and interprets
// the network's output according to the argmax/prune bits in flags:
//
//   flags bit layout:
//     bit 0 : net0 argmax  — Python wrote raw logits; C++ picks legal argmax
//     bit 1 : net1 argmax  — Python wrote softmax probs; C++ normalizes &
//     samples bit 2 : net0 prune   — use SPR-pruned action set (true) or full
//     menu bit 3 : net1 prune
//
// gamePair plays both passes with the same shuffled cards (seats swapped on
// pass B) and appends per-pass payoffs to sched.sbPayoffs / sched.bbPayoffs.
// ---------------------------------------------------------------------------

namespace Match {

Task<float> singleGame(std::array<Card, 9> cards, bool passB, uint8_t flags,
                       Scheduler &sched);

Task<float> gamePair(uint8_t flags, Scheduler &sched);

} // namespace Match
