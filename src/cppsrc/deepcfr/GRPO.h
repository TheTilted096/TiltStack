#include "DeepCFR.h"

namespace GRPO{

Task<float> rollout(CFRGame game, bool hero, Scheduler &sched);

// Recursive worker: takes `game` by reference so the same CFRGame object is
// shared across the entire tree.  makeMove/unmakeMove maintain game state;
// no CFRGame copies are made, avoiding shallow-copy of hand_indexer_t.
// heroReach is the product of all hero action probabilities on the path to
// this node; used to weight samples by state reachability.
Task<float> traverse(CFRGame &game, bool hero, float heroReach, Scheduler &sched);

};