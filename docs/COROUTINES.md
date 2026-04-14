# Multithreaded Coroutine Rollout Machinery

This document explains how TiltStack uses C++20 stackless coroutines to parallelize game rollouts across CPU threads while maintaining high GPU utilization through automatic batching of neural inference requests.

## The Problem: GPU Saturation

Naïve approaches to batching neural inference during game tree traversal face a dilemma:

- **Global batch accumulation** (pause all games, wait for N InfoSets): Forces games into lockstep; slow games delay fast ones.
- **Per-decision queuing** (one network call per decision point): Hundreds of tiny batches; GPU is starved.
- **Fixed batch buffer** (buffer N decisions, process, reset): Games must pause at synchronized points; branch divergence kills CPU efficiency.

**Coroutines solve this**: Each game is a lightweight, suspendable task that yields at neural inference points without blocking its CPU thread.

## Coroutine Basics (C++20)

A coroutine is a function that can suspend and resume. The C++ standard library provides:

- `co_await expr` — suspend the coroutine; expr defines what happens while suspended
- `co_return value` — exit the coroutine and return a value
- `promise_type` — user-defined logic for suspension/resumption

The compiler allocates a **coroutine frame** (heap-allocated state) to hold local variables across suspensions.

### Frame Allocation: The Custom Pool

Naïve coroutine frame allocation hits the heap on every allocation/deallocation. TiltStack uses a **per-thread frame pool** (`Coroutine.h::CoroFramePool`):

```cpp
struct CoroFramePool {
    Node *head = nullptr;  // Free list head
    
    // Allocate CHUNK_SIZE=1024 frames in one OS call
    void *allocate(std::size_t n) {
        if (head) {
            Node *p = head;
            head = head->next;  // Pop from free list (O(1))
            return p;
        }
        // Allocate 1024 frames contiguously
        char *chunk = ::operator new(CHUNK_SIZE * n);
        // Link them and return first
        ...
        return chunk;
    }
    
    // Deallocate by pushing onto free list (O(1))
    void deallocate(void *p, std::size_t n) {
        reinterpret_cast<Node *>(p)->next = head;
        head = p;
    }
};
```

**Benefits:**
- **Amortized O(1)**: 1024 frames allocated at once; each frame is popped/pushed in O(1)
- **Cache locality**: Contiguous slab means frames for the same iteration live close together
- **No locks**: Each thread has its own pool; no contention

## Game Suspension: The InferenceAwaitable

When a game reaches a decision point, it must request neural inference. The `InferenceAwaitable` struct manages this:

```cpp
struct InferenceAwaitable {
    InfoSet input;          // What to send to the network
    Scheduler &sched;       // Reference to the scheduler
    std::size_t index;      // Position in the output buffer
    
    bool await_ready() noexcept { return false; }  // Always suspend
    
    void await_suspend(Handle handle) noexcept {
        // Coroutine is suspending; register with scheduler
        index = sched.enqueueInference(input, handle);
    }
    
    Regrets await_resume() noexcept {
        // Coroutine is resuming; read the result
        return sched.pendingOutputs[index];
    }
};
```

When a coroutine does `co_await InferenceAwaitable{...}`:

1. `await_ready()` returns false → always suspend
2. `await_suspend()` is called; the coroutine handle is registered and the InfoSet is queued
3. Control returns to the caller; other coroutines continue
4. Eventually, the scheduler batches and invokes the network
5. `await_resume()` is called automatically; the coroutine reads its result and continues

## The Scheduler

The scheduler (`Orchestrator` in the codebase) coordinates N game threads and 1 inference thread:

```cpp
class Orchestrator {
    std::vector<Scheduler> schedulers;  // One per worker thread
    InferenceQueue inference_queue;     // Shared: games → inference thread
    
    void run_games(int thread_id) {
        Scheduler &sched = schedulers[thread_id];
        while (not done) {
            // Spawn ~100 coroutines (games), let them suspend
            auto coro = DeepCFR::rollout(game, hero, t, sched);
            coro.resume();  // Run until first suspension
            
            // Process batches until all coroutines finish or suspend
            while (sched.has_pending()) {
                sched.batch_and_send();  // Send to GPU
                sched.receive_results();  // Receive and resume all
            }
        }
    }
    
    void run_inference() {
        while (not done) {
            // Collect requests from all schedulers
            auto batch = inference_queue.dequeue_batch(max_size=1000);
            if (batch.empty()) { sleep(1ms); continue; }
            
            // Dispatch to GPU
            auto output = model.forward(batch);
            
            // Return results to all schedulers
            for (auto &sched : schedulers) {
                sched.fill_results(output);
            }
        }
    }
};
```

## Execution Flow

### Iteration T: P0 Rollout Phase

```
Main thread spawns N worker threads:

Worker Thread 1         Worker Thread 2         Inference Thread
─────────────────────  ─────────────────────  ──────────────────
Spawn 100 coros        Spawn 100 coros
  ↓                      ↓
Game 1 → suspend       Game 101 → suspend
  ↓                      ↓
[batch 100 infos]  +  [batch 100 infos]
        ↓──────────────────→ Model.forward()
                                     ↓
        ←──────────────── [100 logits] + [100 logits]
        ↓                      ↓
Resume → continue      Resume → continue
  ↓                      ↓
Game 1 → suspend       Game 101 → suspend
  (NEXT batch)           (NEXT batch)
  ...
All games finish        All games finish
     ↓                       ↓
Synchronize (wait for all threads)

Collect advantage data from Games 1-200
Train adv_net[0] for 5 epochs
```

### Key Properties

1. **No global synchronization until iteration end**: Games proceed at their own pace; slow games don't hold fast ones
2. **Automatic batching**: Scheduler collects requests from *all* threads, sends one large batch to GPU
3. **High GPU utilization**: Hundreds of games → thousands of inference requests → GPU saturation
4. **Minimal CPU overhead**: Frame allocation is pooled and lock-free; thread overhead is negligible

## Thread Safety

The scheduler **does not use locks** within a single worker thread:

- Each thread owns its own scheduler (frame pool, InfoSet buffer, output buffer)
- No contention; no false sharing
- Inference thread reads from a lock-free queue (CAS-based)

Results are returned via a simple array write — synchronized only at iteration boundaries.

## Coroutine Frame Layout

A coroutine frame for `DeepCFR::traverse()` contains:

```
Frame Header (compiler-managed)
├── Promise object (return value, exception)
├── Coroutine state (current suspension point)
└── Resumption point (for `co_await`)

Frame Body (function locals)
├── CFRGame game (by-reference parameter; ~200 bytes)
├── hero, t (scalar parameters)
├── Scheduler ref
├── Local variables (strategy, regrets, child_ev, etc.)
└── [Suspension frame for InferenceAwaitable]
```

The CFRGame object *is not copied*. A reference to the *parent frame's copy* is captured. This is crucial: it ensures `makeMove()/unmakeMove()` modifies the same game state across all recursive calls.

## Comparison: Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Global batching** | Simple; good GPU utilization | Forces lockstep; slow due to synchronization |
| **Per-decision queuing** | Minimal delay per game | GPU starvation; 100x slower |
| **Coroutines (TiltStack)** | Auto-batching; high utilization; minimal sync | Complex; requires C++20 |
| **Thread pool + blocking** | Standard; easier to reason | Blocked threads waste resources; poor cache behavior |

## Debugging and Profiling

Useful for monitoring coroutine performance:

```cpp
// In Scheduler::batch_and_send()
std::cerr << "Batch size: " << pending_infos.size() << "\n";

// In Orchestrator::run_inference()
auto start = std::chrono::high_resolution_clock::now();
auto output = model.forward(batch);
auto elapsed = std::chrono::high_resolution_clock::now() - start;
std::cerr << "GPU latency: " << elapsed.count() << " ms\n";
```

Key metrics to watch:
- **Batch size trend**: Should be stable
- **GPU latency**: Should be consistent (GPU is saturated if stable)
- **Thread idle time**: Should be <5% (waiting for GPU, not computation)

