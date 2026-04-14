#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <thread>

#include "DeepCFR.h" // InferenceAwaitable, Task, Scheduler, InferenceQueue

// ---------------------------------------------------------------------------
// Coroutines used as test fixtures
// ---------------------------------------------------------------------------

// Immediately returns without touching inference.
static Task<float> trivialTask() { co_return 0.0f; }

// Suspends once for inference, stores the result, then returns.
static Task<float> singleInferenceTask(Scheduler &sched, Regrets &out) {
    out = co_await InferenceAwaitable{InfoSet{}, sched};
    co_return 0.0f;
}

// Same, but writes to an indexed slot — used to verify each coroutine
// reads its own output and not a neighbour's.
static Task<float> indexedInferenceTask(Scheduler &sched, int id,
                                        Regrets *slots) {
    slots[id] = co_await InferenceAwaitable{InfoSet{}, sched};
    co_return 0.0f;
}

// ---------------------------------------------------------------------------
// InferenceQueue
// ---------------------------------------------------------------------------

// Items must come out in the order they went in.
TEST(InferenceQueue, FIFOOrder) {
    InferenceQueue queue, dummy;
    Scheduler s1(dummy), s2(dummy);

    queue.push(&s1);
    queue.push(&s2);

    EXPECT_EQ(queue.pop(), &s1);
    EXPECT_EQ(queue.pop(), &s2);
}

// pop() must block until another thread calls push().
TEST(InferenceQueue, BlocksUntilPush) {
    InferenceQueue queue, dummy;
    Scheduler s(dummy);

    std::atomic<bool> popped{false};
    std::thread t([&]() {
        queue.pop();
        popped = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(popped);

    queue.push(&s);
    t.join();
    EXPECT_TRUE(popped);
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

// A coroutine that immediately co_returns must not cause run() to block.
TEST(Scheduler, TrivialCoroutineTerminates) {
    InferenceQueue queue;
    Scheduler sched(queue);

    sched.spawn(trivialTask());
    sched.run();
}

// A coroutine suspended at inference must resume with the exact values
// written by the "Python" side via outputData() + submitBatch().
TEST(Scheduler, SingleInferenceRound) {
    InferenceQueue queue;
    Scheduler sched(queue);

    Regrets result{};
    sched.spawn(singleInferenceTask(sched, result));

    std::thread worker([&]() { sched.run(); });

    Scheduler *s = queue.pop(); // blocks until flushBatch() pushes
    ASSERT_EQ(s, &sched);
    ASSERT_EQ(s->batchSize(), 1);

    const Regrets expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    *s->outputData() = expected;
    s->submitBatch();
    worker.join();

    EXPECT_EQ(result, expected);
}

// With two coroutines suspended simultaneously, each must read back its own
// output slot — slot 0 and slot 1 must not be swapped.
TEST(Scheduler, IndexStability) {
    InferenceQueue queue;
    Scheduler sched(queue);

    Regrets slots[2]{};
    sched.spawn(indexedInferenceTask(sched, 0, slots));
    sched.spawn(indexedInferenceTask(sched, 1, slots));

    std::thread worker([&]() { sched.run(); });

    Scheduler *s = queue.pop();
    ASSERT_EQ(s->batchSize(), 2);

    const Regrets out0 = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const Regrets out1 = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    s->outputData()[0] = out0;
    s->outputData()[1] = out1;
    s->submitBatch();
    worker.join();

    EXPECT_EQ(slots[0], out0);
    EXPECT_EQ(slots[1], out1);
}
