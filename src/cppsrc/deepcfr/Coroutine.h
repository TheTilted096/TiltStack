#pragma once

#include <coroutine>
#include <exception>
#include <map>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Per-thread coroutine frame pool
//
// Overriding operator new/delete in promise_type routes every frame through
// this pool instead of the global heap.  Freed frames are linked through
// their own memory (the first word becomes a next pointer), so allocation is
// a single pointer pop and deallocation is a single pointer push — no locks,
// no virtual dispatch, no fragmentation.
//
// A separate Bucket is maintained per distinct frame size, so coroutines of
// different sizes (e.g. rollout vs traverse) each get their own free list
// rather than one size locking out the others.
// ---------------------------------------------------------------------------

struct CoroFramePool {
    struct Node {
        Node *next;
    };

    struct Bucket {
        Node *head = nullptr;
        std::vector<void *> chunks;
        int allocCount = 0;
    };

    std::map<std::size_t, Bucket> buckets;

    static constexpr int CHUNK_SIZE = 1024;

    void *allocate(std::size_t n) {
        Bucket &b = buckets[n];
        b.allocCount++;

        if (b.head) {
            Node *p = b.head;
            b.head = b.head->next;
            return p;
        }

        char *chunk = static_cast<char *>(::operator new(CHUNK_SIZE * n));
        b.chunks.push_back(chunk);

        for (int i = 1; i < CHUNK_SIZE - 1; ++i) {
            Node *node = reinterpret_cast<Node *>(chunk + i * n);
            node->next = reinterpret_cast<Node *>(chunk + (i + 1) * n);
        }
        reinterpret_cast<Node *>(chunk + (CHUNK_SIZE - 1) * n)->next = nullptr;

        b.head = reinterpret_cast<Node *>(chunk + n);
        return chunk;
    }

    void deallocate(void *p, std::size_t n) noexcept {
        auto *node = static_cast<Node *>(p);
        node->next = buckets[n].head;
        buckets[n].head = node;
    }

    ~CoroFramePool() {
        for (auto &[size, bucket] : buckets)
            for (void *chunk : bucket.chunks)
                ::operator delete(chunk);
    }

    struct Stats {
        std::map<std::size_t, int> allSizes;
    };

    Stats getStats() const {
        Stats s;
        for (const auto &[sz, b] : buckets)
            s.allSizes[sz] = b.allocCount;
        return s;
    }

    void resetStats() {
        for (auto &[sz, b] : buckets)
            b.allocCount = 0;
    }
};

inline thread_local CoroFramePool coroPool;

// FinalAwaiter: on co_return, symmetric-transfer to the parent that is
// co_await-ing this task, or fall through to noop if this is a root task.
template <typename Promise> struct FinalAwaiter {
    bool await_ready() noexcept { return false; }
    std::coroutine_handle<>
    await_suspend(std::coroutine_handle<Promise> h) noexcept {
        auto cont = h.promise().continuation_;
        return cont ? cont : std::noop_coroutine();
    }
    void await_resume() noexcept {}
};

template <typename T> class Task {
  public:
    struct promise_type {
        T value_{};
        std::coroutine_handle<> continuation_{};

        static void *operator new(std::size_t n) {
            return coroPool.allocate(n);
        }
        static void operator delete(void *p, std::size_t n) noexcept {
            coroPool.deallocate(p, n);
        }

        Task get_return_object() noexcept {
            return Task{
                std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() noexcept { return {}; }
        FinalAwaiter<promise_type> final_suspend() noexcept { return {}; }
        void return_value(T v) noexcept { value_ = std::move(v); }
        void unhandled_exception() noexcept { std::terminate(); }
    };

    explicit Task(std::coroutine_handle<promise_type> h) noexcept
        : handle_(h) {}
    Task(Task &&o) noexcept : handle_(std::exchange(o.handle_, {})) {}
    Task(const Task &) = delete;
    Task &operator=(Task &&) = delete;
    Task &operator=(const Task &) = delete;
    ~Task() {
        if (handle_)
            handle_.destroy();
    }

    bool done() const noexcept { return !handle_ || handle_.done(); }
    std::coroutine_handle<promise_type> handle() const noexcept {
        return handle_;
    }

    // Awaitable — symmetric transfer directly into child, bypassing the
    // scheduler.
    bool await_ready() const noexcept { return done(); }
    std::coroutine_handle<>
    await_suspend(std::coroutine_handle<> caller) noexcept {
        handle_.promise().continuation_ = caller;
        return handle_;
    }
    T await_resume() { return std::move(handle_.promise().value_); }

  private:
    std::coroutine_handle<promise_type> handle_;
};
