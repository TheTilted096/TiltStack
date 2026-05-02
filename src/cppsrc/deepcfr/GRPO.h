#include "DeepCFR.h"

namespace GRPO{

Task<float> rollout(CFRGame game, bool hero, Scheduler &sched);

// Recursive worker: takes `game` by reference so the same CFRGame object is
// shared across the entire tree.  makeMove/unmakeMove maintain game state;
// no CFRGame copies are made, avoiding shallow-copy of hand_indexer_t.
Task<float> traverse(CFRGame &game, bool hero, Scheduler &sched);

};